/* Error-code parity: do failures fail the SAME WAY through a clio connector as
 * they do natively?  (VFD_VOL_TECHNICAL_GOALS.md §1.1(e).)
 *
 * §1.1(e) asks that every HDF5 call return "the same success/failure status and
 * the same output values as under native, including error codes on negative
 * paths".  Every other arm of this suite tests the positive path.  This is the
 * only one that tests the negative path, and it is the axis where the audit
 * found the worst class of defect: a call that FAILED but reported success.
 *
 * WHY THIS EXISTS RATHER THAN A TARGETED REGRESSION TEST.  The swallowed-return
 * defect could not be tested standalone: injecting a specific failure that
 * reaches the connector's read/write path proved impractical.  Comparing
 * against native for *any* operation that fails does not require engineering
 * the failure at all -- which is why this is a differential program, not an
 * assertion program.  It contains no expected values whatsoever.
 *
 * CONTRACT WITH THE DRIVER.  This program is connector-AGNOSTIC on purpose: it
 * includes no clio header, links no clio library, and never asks which
 * connector it is running under.  It just performs operations that must fail
 * and reports how they failed, one line per case:
 *
 *     EPARITY <case> rc=<OK|FAIL> maj=<major msg> min=<minor msg>
 *
 * hdf5_compat_suite.py runs it once natively and once per connector/cache
 * setting and diffs those lines.  It therefore always "passes" on its own
 * (exit 0 unless the harness itself broke); the verdict lives in the diff.
 *
 * WHAT IS COMPARED, AND WHAT IS NOT.  Only the API-level frame of the error
 * stack (H5E_WALK_DOWNWARD's first entry) is reported.  Stack DEPTH is
 * deliberately not compared: a pass-through connector legitimately pushes extra
 * frames, so comparing depth would manufacture failures that mean nothing.  The
 * major/minor are reported as their message STRINGS rather than their hid_t
 * values because those ids are assigned dynamically and are not stable across
 * processes.  Likewise rc is reported as OK/FAIL rather than the raw return:
 * a failing hid_t is just "negative", and its exact value carries no contract.
 */
#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM 64

static char g_dir[512] = "/tmp/hdf5compat";

static void path_for(char *out, size_t n, const char *leaf) {
  snprintf(out, n, "%s/eparity_%s.h5", g_dir, leaf);
}

/* One error-stack frame, flattened to strings. */
typedef struct {
  int have;
  char maj[160];
  char min[160];
} eframe_t;

static herr_t walk_cb(unsigned n, const H5E_error2_t *err, void *client) {
  eframe_t *out = (eframe_t *)client;
  (void)n;
  if (out->have) return 0; /* first frame only -- the API-level one */
  H5E_type_t t;
  if (H5Eget_msg(err->maj_num, &t, out->maj, sizeof(out->maj)) < 0)
    snprintf(out->maj, sizeof(out->maj), "<unreadable>");
  if (H5Eget_msg(err->min_num, &t, out->min, sizeof(out->min)) < 0)
    snprintf(out->min, sizeof(out->min), "<unreadable>");
  out->have = 1;
  return 0;
}

/* Emit one comparable line and reset the stack for the next case. `failed` is
 * the caller's verdict on the return value (nonzero = the call failed). */
static void report(const char *name, int failed) {
  eframe_t e;
  memset(&e, 0, sizeof(e));
  H5Ewalk2(H5E_DEFAULT, H5E_WALK_DOWNWARD, walk_cb, &e);
  printf("EPARITY %s rc=%s maj=%s min=%s\n", name, failed ? "FAIL" : "OK",
         e.have ? e.maj : "-", e.have ? e.min : "-");
  fflush(stdout);
  H5Eclear2(H5E_DEFAULT);
}

/* Build a small int dataset. Returns 0 on success. */
static int make_file(const char *path, int chunked_fletcher) {
  hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (f < 0) return -1;
  hsize_t dims[1] = {DIM};
  hid_t s = H5Screate_simple(1, dims, NULL);
  hid_t dcpl = H5P_DEFAULT;
  if (chunked_fletcher) {
    hsize_t ch[1] = {DIM};
    dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, ch);
    H5Pset_fletcher32(dcpl);
  }
  hid_t d = H5Dcreate2(f, "a", H5T_NATIVE_INT, s, H5P_DEFAULT, dcpl, H5P_DEFAULT);
  int buf[DIM];
  for (int i = 0; i < DIM; i++) buf[i] = 1000 + i;
  herr_t w = H5Dwrite(d, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
  H5Dclose(d);
  H5Sclose(s);
  if (dcpl != H5P_DEFAULT) H5Pclose(dcpl);
  H5Fclose(f);
  return (w < 0) ? -1 : 0;
}

/* Flip a bit inside the dataset's RAW DATA, by locating the byte pattern we
 * know we wrote. Works without parsing the file format: fletcher32 is a
 * checksum, not a compressor, so the ints are present verbatim. Returns 0 if a
 * byte was flipped. */
static int corrupt_raw(const char *path) {
  FILE *fh = fopen(path, "r+b");
  if (!fh) return -1;
  fseek(fh, 0, SEEK_END);
  long n = ftell(fh);
  fseek(fh, 0, SEEK_SET);
  unsigned char *img = (unsigned char *)malloc((size_t)n);
  if (!img) { fclose(fh); return -1; }
  if (fread(img, 1, (size_t)n, fh) != (size_t)n) { free(img); fclose(fh); return -1; }
  /* Four consecutive values is enough to be unambiguous in a 64-elem file. */
  int probe[4] = {1000, 1001, 1002, 1003};
  long at = -1;
  for (long i = 0; i + (long)sizeof(probe) <= n; i++) {
    if (memcmp(img + i, probe, sizeof(probe)) == 0) { at = i; break; }
  }
  free(img);
  if (at < 0) { fclose(fh); return -1; }
  fseek(fh, at, SEEK_SET);
  unsigned char b;
  fread(&b, 1, 1, fh);
  b ^= 0xFF;
  fseek(fh, at, SEEK_SET);
  fwrite(&b, 1, 1, fh);
  fclose(fh);
  return 0;
}

/* ------------------------------------------------------------------ cases */

static void case_open_missing(void) {
  char p[600];
  path_for(p, sizeof(p), "nonexistent_XYZ");
  remove(p);
  hid_t f = H5Fopen(p, H5F_ACC_RDONLY, H5P_DEFAULT);
  report("open_missing_file", f < 0);
  if (f >= 0) H5Fclose(f);
}

static void case_create_dup_dataset(void) {
  char p[600];
  path_for(p, sizeof(p), "dup");
  if (make_file(p, 0) < 0) { report("create_duplicate_dataset", -1); return; }
  hid_t f = H5Fopen(p, H5F_ACC_RDWR, H5P_DEFAULT);
  hsize_t dims[1] = {DIM};
  hid_t s = H5Screate_simple(1, dims, NULL);
  hid_t d = H5Dcreate2(f, "a", H5T_NATIVE_INT, s, H5P_DEFAULT, H5P_DEFAULT,
                       H5P_DEFAULT); /* "a" already exists */
  report("create_duplicate_dataset", d < 0);
  if (d >= 0) H5Dclose(d);
  H5Sclose(s);
  H5Fclose(f);
}

static void case_open_missing_dataset(void) {
  char p[600];
  path_for(p, sizeof(p), "missing_dset");
  if (make_file(p, 0) < 0) { report("open_missing_dataset", -1); return; }
  hid_t f = H5Fopen(p, H5F_ACC_RDONLY, H5P_DEFAULT);
  hid_t d = H5Dopen2(f, "not_there", H5P_DEFAULT);
  report("open_missing_dataset", d < 0);
  if (d >= 0) H5Dclose(d);
  H5Fclose(f);
}

static void case_write_readonly(void) {
  char p[600];
  path_for(p, sizeof(p), "readonly");
  if (make_file(p, 0) < 0) { report("write_to_readonly_file", -1); return; }
  hid_t f = H5Fopen(p, H5F_ACC_RDONLY, H5P_DEFAULT);
  hid_t d = H5Dopen2(f, "a", H5P_DEFAULT);
  int buf[DIM];
  for (int i = 0; i < DIM; i++) buf[i] = 7;
  herr_t w = H5Dwrite(d, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
  report("write_to_readonly_file", w < 0);
  H5Dclose(d);
  H5Fclose(f);
}

static void case_selection_mismatch(void) {
  char p[600];
  path_for(p, sizeof(p), "selmismatch");
  if (make_file(p, 0) < 0) { report("selection_count_mismatch", -1); return; }
  hid_t f = H5Fopen(p, H5F_ACC_RDONLY, H5P_DEFAULT);
  hid_t d = H5Dopen2(f, "a", H5P_DEFAULT);
  hid_t fs = H5Dget_space(d);
  hsize_t start[1] = {0}, count[1] = {10};
  H5Sselect_hyperslab(fs, H5S_SELECT_SET, start, NULL, count, NULL);
  hsize_t mdim[1] = {5}; /* 5 != 10 -- element counts must match */
  hid_t ms = H5Screate_simple(1, mdim, NULL);
  int buf[DIM];
  herr_t r = H5Dread(d, H5T_NATIVE_INT, ms, fs, H5P_DEFAULT, buf);
  report("selection_count_mismatch", r < 0);
  H5Sclose(ms);
  H5Sclose(fs);
  H5Dclose(d);
  H5Fclose(f);
}

static void case_bad_conversion(void) {
  char p[600];
  path_for(p, sizeof(p), "badconv");
  if (make_file(p, 0) < 0) { report("incompatible_type_conversion", -1); return; }
  hid_t f = H5Fopen(p, H5F_ACC_RDONLY, H5P_DEFAULT);
  hid_t d = H5Dopen2(f, "a", H5P_DEFAULT);
  /* No conversion path exists from a file integer to a memory compound. */
  hid_t ct = H5Tcreate(H5T_COMPOUND, sizeof(double));
  H5Tinsert(ct, "x", 0, H5T_NATIVE_DOUBLE);
  double buf[DIM];
  herr_t r = H5Dread(d, ct, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
  report("incompatible_type_conversion", r < 0);
  H5Tclose(ct);
  H5Dclose(d);
  H5Fclose(f);
}

/* The one case whose whole point is the CACHE.
 *
 * The file is written, closed, damaged on disk behind HDF5's back, then
 * reopened and read. Natively the fletcher32 filter catches it and the read
 * fails. Through a connector holding a staged copy of the data, the read can
 * SUCCEED from that copy -- the cache masking the file's own error. That is a
 * known-open gap (external-modification staleness), recorded in the VOL README,
 * and it is exactly why this program is run in both cache-on and cache-off
 * modes: cache-off is where error parity is meaningful, cache-on is where the
 * masking becomes visible as a parity difference instead of staying invisible.
 */
static void case_checksum(void) {
  char p[600];
  path_for(p, sizeof(p), "checksum");
  if (make_file(p, 1) < 0) { report("corrupt_checksum_read", -1); return; }
  /* Read once before corrupting: this is what gives a cache something to hold,
     so the masking behaviour is actually reachable rather than hypothetical. */
  hid_t f0 = H5Fopen(p, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (f0 >= 0) {
    hid_t d0 = H5Dopen2(f0, "a", H5P_DEFAULT);
    if (d0 >= 0) {
      int warm[DIM];
      H5Dread(d0, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, warm);
      H5Dclose(d0);
    }
    H5Fclose(f0);
  }
  H5Eclear2(H5E_DEFAULT);
  if (corrupt_raw(p) < 0) { report("corrupt_checksum_read", -1); return; }
  hid_t f = H5Fopen(p, H5F_ACC_RDONLY, H5P_DEFAULT);
  hid_t d = H5Dopen2(f, "a", H5P_DEFAULT);
  int buf[DIM];
  herr_t r = H5Dread(d, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
  report("corrupt_checksum_read", r < 0);
  if (d >= 0) H5Dclose(d);
  if (f >= 0) H5Fclose(f);
}

int main(int argc, char **argv) {
  if (argc > 1) snprintf(g_dir, sizeof(g_dir), "%s", argv[1]);
  /* The stack is read programmatically; the automatic dump would only pollute
     the output the driver parses. */
  H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

  case_open_missing();
  case_create_dup_dataset();
  case_open_missing_dataset();
  case_write_readonly();
  case_selection_mismatch();
  case_bad_conversion();
  case_checksum();

  printf("error_parity: emitted\n");
  return 0;
}
