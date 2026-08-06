/* Library-free readability: can a reader that does NOT link libhdf5 get the data
 * out of a connector-produced file using only POSIX I/O?
 * (VFD_VOL_TECHNICAL_GOALS.md §1.1(d).)
 *
 * WHY THIS IS THE CRITERION THAT MATTERS MOST.  (a)-(c) and (e) all ask HDF5
 * whether HDF5 likes the file.  They cannot distinguish "the bytes are in the
 * file" from "the bytes are somewhere CLIO can find them", because the library
 * is in the loop either way.  This one removes the library: it parses the
 * superblock, walks the group graph, finds the dataset, reads the raw data
 * through pread(2), and never calls a single HDF5 function.  It is the check
 * that "most cleanly falsifies any design that keeps data outside the native
 * file", and it is what HDF5 ToolsUI actually requires.  Before this, §1.1(d)
 * was an 8-byte superblock signature probe, which proves only that the file
 * starts with the right magic number.
 *
 * NO HDF5 HEADER IS INCLUDED HERE, DELIBERATELY.  Not hdf5.h, not H5public.h,
 * nothing.  If this file ever grows an HDF5 include it stops testing what it
 * claims to test.  The only file I/O primitives used are open/pread/close --
 * no mmap, no stdio on the target file, no seek-and-read dance that could
 * accidentally rely on library buffering.
 *
 * OUTPUT (one line per requested dataset; the driver diffs these against a
 * native h5py read of the same dataset):
 *
 *   LIBFREE <dset> ok    class=<contiguous|compact> nbytes=<n> fnv=<hex>
 *   LIBFREE <dset> skip  reason=<why this dataset is out of scope>
 *   LIBFREE <dset> fail  reason=<why the parse failed>
 *
 * SCOPE, STATED UP FRONT RATHER THAN DISCOVERED.  This reads superblock v0 with
 * old-style (symbol-table + local-heap) groups, object header v1, and
 * contiguous or compact unfiltered layouts.  That is what the suite's corpus
 * actually contains (verified with h5dump -B/-p), not a guess.  Explicitly NOT
 * supported, each reported as a `skip` with its reason rather than silently
 * passing:
 *
 *   - chunked layouts      -- needs the chunk B-tree index walk
 *   - filtered datasets    -- the raw bytes are shuffled/checksummed, so
 *                             comparing them to a native read is meaningless
 *                             without reimplementing the filter pipeline
 *   - variable-length data -- the contiguous bytes are heap references, not the
 *                             values; following them is a separate mechanism
 *   - external / virtual   -- the raw data is in ANOTHER file BY DESIGN. That is
 *                             HDF5 semantics, not a CLIO artifact, so a "cannot
 *                             read it here" result would be a false alarm.
 *
 * A skip is not a pass. The driver counts verified datasets and reports the
 * skips, so the coverage of this gate is visible instead of implied.
 */
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define SB_SIG "\211HDF\r\n\032\n"
#define MAX_DIMS 32

typedef struct {
  int fd;
  unsigned osize; /* size of offsets  */
  unsigned lsize; /* size of lengths  */
  uint64_t root_oh;
} h5f_t;

/* ------------------------------------------------------------------ raw I/O */
/* Every byte this program learns about the file comes through here. */
static int rd(const h5f_t *f, uint64_t off, void *buf, size_t n) {
  size_t done = 0;
  while (done < n) {
    ssize_t got = pread(f->fd, (char *)buf + done, n - done, (off_t)(off + done));
    if (got <= 0) return -1;
    done += (size_t)got;
  }
  return 0;
}

static uint64_t le(const uint8_t *p, unsigned n) {
  uint64_t v = 0;
  for (unsigned i = 0; i < n; i++) v |= (uint64_t)p[i] << (8 * i);
  return v;
}

/* Undefined address ("this thing isn't here") is all-1s in the offset width. */
static int undef_addr(uint64_t a, unsigned osize) {
  return a == (osize >= 8 ? ~(uint64_t)0 : ((uint64_t)1 << (8 * osize)) - 1);
}

/* 64-bit FNV-1a. Not a security hash -- it only has to let the driver compare
 * these bytes against the same bytes read via h5py, and the driver computes it
 * the same way in Python. */
static uint64_t fnv1a(const uint8_t *p, size_t n, uint64_t h) {
  for (size_t i = 0; i < n; i++) {
    h ^= p[i];
    h *= 0x100000001b3ULL;
  }
  return h;
}

/* ------------------------------------------------------------- superblock */
static int open_h5(h5f_t *f, const char *path) {
  uint8_t sb[64];
  f->fd = open(path, O_RDONLY);
  if (f->fd < 0) return -1;
  if (rd(f, 0, sb, sizeof(sb)) < 0) return -1;
  if (memcmp(sb, SB_SIG, 8) != 0) return -2;
  if (sb[8] != 0) return -3; /* only superblock v0 is in scope */
  f->osize = sb[13];
  f->lsize = sb[14];
  if (f->osize > 8 || f->lsize > 8) return -4;
  /* Root group symbol table entry begins at 56: link name offset (lsize) then
     the object header address (osize). */
  uint8_t buf[16];
  if (rd(f, 56 + f->lsize, buf, f->osize) < 0) return -1;
  f->root_oh = le(buf, f->osize);
  return 0;
}

/* --------------------------------------------------- object header v1 walk */
/* What we care about out of an object header. */
typedef struct {
  int have_stab, have_layout, have_dtype, have_space;
  int has_filters, has_external;
  uint64_t stab_btree, stab_heap;
  unsigned layout_class; /* 0 compact, 1 contiguous, 2 chunked, 3 virtual */
  uint64_t data_addr, data_size;
  uint64_t compact_at, compact_size;
  unsigned dtype_class, dtype_size, dtype_bits;
  uint64_t nelem;
  unsigned rank;               /* dataspace rank                            */
  uint64_t dims[MAX_DIMS];     /* dataspace dims                            */
  unsigned chunk_rank;         /* == rank; the stored value is rank+1       */
  uint64_t chunk_dims[MAX_DIMS];
  uint64_t chunk_esize;        /* trailing "dimension" is the element size  */
  uint64_t btree_addr;         /* chunk index                               */
} oh_t;

static int walk_oh_block(const h5f_t *f, uint64_t start, uint64_t end, oh_t *o,
                         int depth);

static int oh_message(const h5f_t *f, unsigned type, uint64_t data,
                      uint64_t size, oh_t *o, int depth) {
  uint8_t b[256];
  switch (type) {
    case 0x0001: { /* Dataspace */
      if (rd(f, data, b, 8) < 0) return -1;
      unsigned ndims = b[1];
      if (ndims > MAX_DIMS) return -1;
      uint64_t n = 1;
      for (unsigned i = 0; i < ndims; i++) {
        uint8_t d[8];
        if (rd(f, data + 8 + (uint64_t)i * f->lsize, d, f->lsize) < 0) return -1;
        o->dims[i] = le(d, f->lsize);
        n *= o->dims[i];
      }
      o->rank = ndims;
      o->nelem = n; /* ndims==0 is a scalar: the empty product, 1 */
      o->have_space = 1;
      break;
    }
    case 0x0003: { /* Datatype */
      if (rd(f, data, b, 8) < 0) return -1;
      /* Byte 0 packs BOTH fields: version in the upper 4 bits, class in the
         LOWER 4. Getting this backwards is quiet rather than loud -- the size
         field still parses correctly, so a vlen string reads as a plausible
         48-byte contiguous dataset and only the hash comparison against a
         native read would ever notice. */
      o->dtype_class = (unsigned)(b[0] & 0x0F);
      o->dtype_size = (unsigned)le(b + 4, 4);
      /* Class bit field, bytes 1..3. For class 3 (string) bits 0-3 are the
         padding type and 4-7 the character set; NEITHER says "variable". A
         variable-length string is encoded as a VLEN whose bit field marks it a
         string, so keep the bit field to tell those apart below. */
      o->dtype_bits = (unsigned)le(b + 1, 3);
      o->have_dtype = 1;
      if (getenv("CLIO_LIBFREE_DEBUG"))
        fprintf(stderr, "[libfree] dtype class=%u size=%u bits=0x%06x\n",
                o->dtype_class, o->dtype_size, o->dtype_bits);
      break;
    }
    case 0x0007: /* External data files -- raw data lives outside this file */
      o->has_external = 1;
      break;
    case 0x000B: /* Filter pipeline */
      o->has_filters = 1;
      break;
    case 0x0008: { /* Data layout */
      if (rd(f, data, b, 2) < 0) return -1;
      unsigned ver = b[0];
      if (ver != 3 && ver != 4) return -1;
      o->layout_class = b[1];
      if (o->layout_class == 1) { /* contiguous: address, size */
        if (rd(f, data + 2, b, f->osize + f->lsize) < 0) return -1;
        o->data_addr = le(b, f->osize);
        o->data_size = le(b + f->osize, f->lsize);
      } else if (o->layout_class == 0) { /* compact: size, then inline data */
        if (rd(f, data + 2, b, 2) < 0) return -1;
        o->compact_size = le(b, 2);
        o->compact_at = data + 4;
      } else if (o->layout_class == 2) {
        /* chunked: dimensionality(1), b-tree address(osize), then
           `dimensionality` 4-byte values. The stored dimensionality is the
           dataset rank PLUS ONE -- the trailing entry is the element size, not
           a dimension. Treating it as one is a classic off-by-one here. */
        if (rd(f, data + 2, b, 1) < 0) return -1;
        unsigned stored = b[0];
        if (stored == 0 || stored > MAX_DIMS) return -1;
        o->chunk_rank = stored - 1;
        if (rd(f, data + 3, b, f->osize) < 0) return -1;
        o->btree_addr = le(b, f->osize);
        for (unsigned i = 0; i < stored; i++) {
          uint8_t d[4];
          if (rd(f, data + 3 + f->osize + (uint64_t)i * 4, d, 4) < 0) return -1;
          uint64_t v = le(d, 4);
          if (i + 1 == stored)
            o->chunk_esize = v;
          else
            o->chunk_dims[i] = v;
        }
      }
      o->have_layout = 1;
      break;
    }
    case 0x0011: { /* Symbol table (old-style group) */
      if (rd(f, data, b, 2 * (uint64_t)f->osize) < 0) return -1;
      o->stab_btree = le(b, f->osize);
      o->stab_heap = le(b + f->osize, f->osize);
      o->have_stab = 1;
      break;
    }
    case 0x0010: { /* Object header continuation */
      if (rd(f, data, b, (uint64_t)f->osize + f->lsize) < 0) return -1;
      uint64_t caddr = le(b, f->osize);
      uint64_t clen = le(b + f->osize, f->lsize);
      if (depth < 8 && !undef_addr(caddr, f->osize))
        return walk_oh_block(f, caddr, caddr + clen, o, depth + 1);
      break;
    }
    default:
      break; /* every other message is irrelevant here */
  }
  (void)size;
  return 0;
}

static int walk_oh_block(const h5f_t *f, uint64_t start, uint64_t end, oh_t *o,
                         int depth) {
  uint64_t cur = start;
  while (cur + 8 <= end) {
    uint8_t mh[8];
    if (rd(f, cur, mh, 8) < 0) return -1;
    unsigned type = (unsigned)le(mh, 2);
    uint64_t msize = le(mh + 2, 2);
    if (oh_message(f, type, cur + 8, msize, o, depth) < 0) return -1;
    cur += 8 + msize;
  }
  return 0;
}

static int read_oh(const h5f_t *f, uint64_t addr, oh_t *o) {
  uint8_t pre[16];
  memset(o, 0, sizeof(*o));
  if (rd(f, addr, pre, 16) < 0) return -1;
  if (pre[0] != 1) return -2; /* only object header v1 is in scope */
  uint64_t hdrsize = le(pre + 8, 4);
  /* Messages start 16 bytes in (12-byte prefix padded to an 8-byte boundary). */
  return walk_oh_block(f, addr + 16, addr + 16 + hdrsize, o, 0);
}

/* ------------------------------------------------------- group name lookup */
/* Read a null-terminated name out of the group's local heap. */
static int heap_name(const h5f_t *f, uint64_t heap_addr, uint64_t name_off,
                     char *out, size_t outn) {
  uint8_t h[32];
  if (rd(f, heap_addr, h, 8) < 0) return -1;
  if (memcmp(h, "HEAP", 4) != 0) return -1;
  /* HEAP: sig(4) ver(1) rsv(3) dseg_size(lsize) freelist(lsize) dseg_addr(osize) */
  uint64_t off = 8 + 2ULL * f->lsize;
  if (rd(f, heap_addr + off, h, f->osize) < 0) return -1;
  uint64_t dseg = le(h, f->osize);
  memset(out, 0, outn);
  for (size_t i = 0; i + 1 < outn; i++) {
    uint8_t c;
    if (rd(f, dseg + name_off + i, &c, 1) < 0) return -1;
    out[i] = (char)c;
    if (c == 0) break;
  }
  return 0;
}

/* Scan one SNOD for `want`; on a hit store the object header address and the
 * entry's cache type. Cache type 2 means a SYMBOLIC (soft) link, and for those
 * the object-header-address field is not an address at all -- it is meaningless,
 * and the link target lives in the local heap. Reading it as an address parses
 * garbage, so the caller has to know the difference. */
static int snod_find(const h5f_t *f, uint64_t addr, uint64_t heap_addr,
                     const char *want, uint64_t *found, unsigned *ctype) {
  uint8_t hdr[8];
  if (rd(f, addr, hdr, 8) < 0) return -1;
  if (memcmp(hdr, "SNOD", 4) != 0) return -1;
  unsigned nsyms = (unsigned)le(hdr + 6, 2);
  uint64_t esize = (uint64_t)f->lsize + f->osize + 4 + 4 + 16;
  for (unsigned i = 0; i < nsyms; i++) {
    uint8_t e[64];
    uint64_t at = addr + 8 + (uint64_t)i * esize;
    if (rd(f, at, e, (size_t)(f->lsize + f->osize + 4)) < 0) return -1;
    uint64_t noff = le(e, f->lsize);
    uint64_t oh = le(e + f->lsize, f->osize);
    unsigned ct = (unsigned)le(e + f->lsize + f->osize, 4);
    char nm[256];
    if (heap_name(f, heap_addr, noff, nm, sizeof(nm)) < 0) continue;
    if (strcmp(nm, want) == 0) {
      *found = oh;
      *ctype = ct;
      return 1;
    }
  }
  return 0;
}

/* Descend a v1 B-tree of group entries, visiting every leaf. The corpus groups
 * are tiny, so an exhaustive walk is simpler and no slower than a keyed search. */
static int btree_find(const h5f_t *f, uint64_t addr, uint64_t heap_addr,
                      const char *want, uint64_t *found, unsigned *ctype,
                      int depth) {
  if (depth > 16 || undef_addr(addr, f->osize)) return 0;
  uint8_t hdr[8];
  if (rd(f, addr, hdr, 8) < 0) return -1;
  if (memcmp(hdr, "TREE", 4) != 0) return -1;
  unsigned level = hdr[5];
  unsigned nused = (unsigned)le(hdr + 6, 2);
  /* sig(4) type(1) level(1) nused(2) left(osize) right(osize), then
     key(lsize) child(osize) repeating, with a trailing key. */
  uint64_t base = 8 + 2ULL * f->osize;
  for (unsigned i = 0; i < nused; i++) {
    uint8_t c[8];
    uint64_t at = base + f->lsize + (uint64_t)i * (f->lsize + f->osize);
    if (rd(f, addr + at, c, f->osize) < 0) return -1;
    uint64_t child = le(c, f->osize);
    int r = (level == 0)
                ? snod_find(f, child, heap_addr, want, found, ctype)
                : btree_find(f, child, heap_addr, want, found, ctype, depth + 1);
    if (r != 0) return r;
  }
  return 0;
}

/* Resolve a "/a/b/c" style path to an object header address. */
static int resolve(const h5f_t *f, const char *path, uint64_t *oh_out) {
  char buf[512];
  snprintf(buf, sizeof(buf), "%s", path);
  uint64_t cur = f->root_oh;
  char *save = NULL;
  for (char *tok = strtok_r(buf, "/", &save); tok;
       tok = strtok_r(NULL, "/", &save)) {
    oh_t o;
    if (read_oh(f, cur, &o) < 0) return -1;
    if (!o.have_stab) return -2; /* not an old-style group */
    uint64_t next = 0;
    unsigned ctype = 0;
    int r = btree_find(f, o.stab_btree, o.stab_heap, tok, &next, &ctype, 0);
    if (r <= 0) return -3; /* name not found */
    if (ctype == 2) return -4; /* soft link: no object header to follow */
    cur = next;
  }
  *oh_out = cur;
  return 0;
}

/* ------------------------------------------------------- chunked reassembly */
/* Walk the v1 chunk B-tree and scatter each chunk into `out`.
 *
 * Node type 1 (chunks). Each key is: chunk byte size (4), filter mask (4), then
 * rank+1 8-byte element offsets -- the trailing one corresponds to the element-
 * size pseudo-dimension and is always 0. Keys and children alternate, with one
 * more key than children.
 *
 * Chunks may hang off the end of the dataset (HDF5 does not trim them), so every
 * copy is clipped to the dataset bounds. Regions with no chunk stay as calloc'd
 * zeros, which is what HDF5's default fill value would give.
 */
static int chunk_tree(const h5f_t *f, const oh_t *o, uint64_t addr,
                      uint8_t *out, uint64_t outbytes, int depth) {
  if (depth > 24 || undef_addr(addr, f->osize)) return 0;
  uint8_t hdr[8];
  if (rd(f, addr, hdr, 8) < 0) return -1;
  if (memcmp(hdr, "TREE", 4) != 0) return -1;
  unsigned level = hdr[5];
  unsigned nused = (unsigned)le(hdr + 6, 2);
  unsigned rank = o->chunk_rank;
  uint64_t keysz = 4 + 4 + 8ULL * (rank + 1);
  uint64_t base = 8 + 2ULL * f->osize;

  /* Row-major element strides for the destination. */
  uint64_t stride[MAX_DIMS];
  uint64_t acc = 1;
  for (int i = (int)rank - 1; i >= 0; i--) {
    stride[i] = acc;
    acc *= o->dims[i];
  }

  for (unsigned i = 0; i < nused; i++) {
    uint8_t key[8 + 8 * MAX_DIMS];
    uint64_t kat = addr + base + (uint64_t)i * (keysz + f->osize);
    if (keysz > sizeof(key)) return -1;
    if (rd(f, kat, key, (size_t)keysz) < 0) return -1;
    uint8_t cb[8];
    if (rd(f, kat + keysz, cb, f->osize) < 0) return -1;
    uint64_t child = le(cb, f->osize);
    if (level > 0) {
      if (chunk_tree(f, o, child, out, outbytes, depth + 1) < 0) return -1;
      continue;
    }
    uint64_t csize = le(key, 4);
    uint64_t off[MAX_DIMS];
    for (unsigned d = 0; d < rank; d++) off[d] = le(key + 8 + 8ULL * d, 8);
    if (undef_addr(child, f->osize) || csize == 0) continue;

    uint8_t *cbuf = (uint8_t *)malloc((size_t)csize);
    if (!cbuf) return -1;
    if (rd(f, child, cbuf, (size_t)csize) < 0) { free(cbuf); return -1; }

    /* Iterate every row of the chunk along the fastest dimension. */
    uint64_t rows = 1;
    for (unsigned d = 0; d + 1 < rank; d++) rows *= o->chunk_dims[d];
    uint64_t lastc = rank ? o->chunk_dims[rank - 1] : 1;
    uint64_t esz = o->chunk_esize;
    for (uint64_t r = 0; r < rows; r++) {
      /* Decompose r into per-dimension indices within the chunk. */
      uint64_t idx[MAX_DIMS], rem = r;
      int oob = 0;
      for (int d = (int)rank - 2; d >= 0; d--) {
        idx[d] = rem % o->chunk_dims[d];
        rem /= o->chunk_dims[d];
        if (off[d] + idx[d] >= o->dims[d]) oob = 1;
      }
      if (oob) continue; /* this row is entirely past the dataset edge */
      uint64_t dst_elem = 0;
      for (unsigned d = 0; d + 1 < rank; d++)
        dst_elem += (off[d] + idx[d]) * stride[d];
      dst_elem += rank ? off[rank - 1] * stride[rank - 1] : 0;
      /* Clip the contiguous run to the dataset's extent. */
      uint64_t avail = rank ? (o->dims[rank - 1] > off[rank - 1]
                                   ? o->dims[rank - 1] - off[rank - 1]
                                   : 0)
                            : 1;
      uint64_t n = lastc < avail ? lastc : avail;
      if (!n) continue;
      uint64_t src_off = r * lastc * esz;
      if (src_off + n * esz > csize) { free(cbuf); return -1; }
      if ((dst_elem + n) * esz > outbytes) { free(cbuf); return -1; }
      memcpy(out + dst_elem * esz, cbuf + src_off, (size_t)(n * esz));
    }
    free(cbuf);
  }
  return 0;
}

/* ------------------------------------------------------------------- main */
static void emit_skip(const char *d, const char *why) {
  printf("LIBFREE %s skip reason=%s\n", d, why);
}

static int one_dataset(const h5f_t *f, const char *dpath) {
  uint64_t oh;
  int rr = resolve(f, dpath, &oh);
  if (rr == -4) {
    /* Not a failure: following a soft link means resolving a path stored in the
       local heap, which is a separate mechanism from reading a dataset's bytes.
       The link's TARGET is covered by this gate under its own name. */
    emit_skip(dpath, "soft_link");
    return 0;
  }
  if (rr < 0) {
    printf("LIBFREE %s fail reason=could_not_resolve_path_rc_%d\n", dpath, rr);
    return 0;
  }
  oh_t o;
  if (read_oh(f, oh, &o) < 0) {
    printf("LIBFREE %s fail reason=object_header_parse\n", dpath);
    return 0;
  }
  if (!o.have_layout || !o.have_dtype || !o.have_space) {
    printf("LIBFREE %s fail reason=missing_layout_dtype_or_space\n", dpath);
    return 0;
  }
  /* Out-of-scope cases, each named. See the scope note at the top.
     has_filters is tested BEFORE the chunked branch on purpose: a filtered
     chunked dataset must still skip, because its stored bytes are
     shuffled/compressed and reassembling them without running the filter
     pipeline would produce confident nonsense. */
  if (o.has_external) { emit_skip(dpath, "external_raw_data"); return 0; }
  if (o.layout_class == 3) { emit_skip(dpath, "virtual_layout"); return 0; }
  if (o.has_filters) { emit_skip(dpath, "filtered"); return 0; }
  if (o.dtype_class == 9) { emit_skip(dpath, "vlen_datatype"); return 0; }

  uint64_t nbytes = o.nelem * (uint64_t)o.dtype_size;

  if (o.layout_class == 2) { /* chunked: reassemble from the chunk B-tree */
    if (o.chunk_rank != o.rank) { emit_skip(dpath, "chunk_rank_mismatch"); return 0; }
    if (undef_addr(o.btree_addr, f->osize)) {
      emit_skip(dpath, "unallocated_chunk_index");
      return 0;
    }
    if (nbytes == 0 || nbytes > (256ULL << 20)) {
      emit_skip(dpath, "chunked_dataset_too_large");
      return 0;
    }
    uint8_t *out = (uint8_t *)calloc(1, (size_t)nbytes);
    if (!out) { printf("LIBFREE %s fail reason=oom\n", dpath); return 0; }
    if (chunk_tree(f, &o, o.btree_addr, out, nbytes, 0) < 0) {
      free(out);
      printf("LIBFREE %s fail reason=chunk_btree_walk\n", dpath);
      return 0;
    }
    uint64_t h = fnv1a(out, (size_t)nbytes, 0xcbf29ce484222325ULL);
    free(out);
    printf("LIBFREE %s ok class=chunked nbytes=%llu fnv=%016llx\n", dpath,
           (unsigned long long)nbytes, (unsigned long long)h);
    return 1;
  }

  uint64_t src;
  const char *cls;
  if (o.layout_class == 1) {
    if (undef_addr(o.data_addr, f->osize)) {
      emit_skip(dpath, "unallocated_contiguous_storage");
      return 0;
    }
    src = o.data_addr;
    cls = "contiguous";
  } else if (o.layout_class == 0) {
    src = o.compact_at;
    cls = "compact";
    if (o.compact_size < nbytes) nbytes = o.compact_size;
  } else {
    emit_skip(dpath, "unknown_layout_class");
    return 0;
  }

  /* The actual point of the program: pull the bytes out with pread and hash
     them, with no HDF5 code anywhere in the process. */
  uint64_t h = 0xcbf29ce484222325ULL;
  uint8_t chunk[8192];
  uint64_t left = nbytes;
  uint64_t at = src;
  while (left) {
    size_t want = left > sizeof(chunk) ? sizeof(chunk) : (size_t)left;
    if (rd(f, at, chunk, want) < 0) {
      printf("LIBFREE %s fail reason=pread_short_at_%llu\n", dpath,
             (unsigned long long)at);
      return 0;
    }
    h = fnv1a(chunk, want, h);
    at += want;
    left -= want;
  }
  printf("LIBFREE %s ok class=%s nbytes=%llu fnv=%016llx\n", dpath, cls,
         (unsigned long long)nbytes, (unsigned long long)h);
  return 1;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <file.h5> <dset> [<dset>...]\n", argv[0]);
    return 2;
  }
  h5f_t f;
  int rc = open_h5(&f, argv[1]);
  if (rc < 0) {
    printf("LIBFREE - fail reason=superblock_rc_%d\n", rc);
    return 1;
  }
  for (int i = 2; i < argc; i++) one_dataset(&f, argv[i]);
  close(f.fd);
  printf("libfree_reader: emitted\n");
  return 0;
}
