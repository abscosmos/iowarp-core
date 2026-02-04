#include <wrp_cae/core/factory/assimilator_factory.h>
#include <wrp_cae/core/factory/binary_file_assimilator.h>
#ifdef WRP_CAE_ENABLE_HDF5
#include <wrp_cae/core/factory/hdf5_file_assimilator.h>
#endif
#ifdef CAE_ENABLE_GLOBUS
#include <wrp_cae/core/factory/globus_file_assimilator.h>
#endif
#include <chimaera/chimaera.h>

#include <memory>

namespace wrp_cae::core {

AssimilatorFactory::AssimilatorFactory(
    std::shared_ptr<wrp_cte::core::Client> cte_client)
    : cte_client_(cte_client) {}

std::unique_ptr<BaseAssimilator> AssimilatorFactory::Get(
    const std::string& src) {
  HLOG(kDebug, "AssimilatorFactory::Get ENTRY: src='{}'", src);

  // Check if this is a Globus web URL first
  if (src.find("https://app.globus.org") == 0) {
#ifdef CAE_ENABLE_GLOBUS
    HLOG(kDebug,
         "AssimilatorFactory: Detected Globus web URL, creating "
         "GlobusFileAssimilator");
    return std::make_unique<GlobusFileAssimilator>(cte_client_);
#else
    HLOG(kError,
         "AssimilatorFactory: Globus web URL detected but Globus support not "
         "compiled in. "
         "Rebuild with -DCAE_ENABLE_GLOBUS=ON to enable Globus support.");
    return nullptr;
#endif
  }

  std::string protocol = GetUrlProtocol(src);
  HLOG(kDebug, "AssimilatorFactory: Extracted protocol='{}'", protocol);

  if (protocol == "file") {
    HLOG(kDebug,
         "AssimilatorFactory: Creating BinaryFileAssimilator for 'file' "
         "protocol");
    // For file protocol, return a BinaryFileAssimilator
    return std::make_unique<BinaryFileAssimilator>(cte_client_);
  } else if (protocol == "hdf5") {
#ifdef WRP_CAE_ENABLE_HDF5
    HLOG(
        kInfo,
        "AssimilatorFactory: Creating Hdf5FileAssimilator for 'hdf5' protocol");
    // For hdf5 protocol, return an Hdf5FileAssimilator
    return std::make_unique<Hdf5FileAssimilator>(cte_client_);
#else
    // HDF5 support not compiled in
    HLOG(kError,
         "AssimilatorFactory: HDF5 protocol requested but HDF5 support not "
         "compiled in. "
         "Rebuild with -DWRP_CORE_ENABLE_HDF5=ON to enable HDF5 support.");
    return nullptr;
#endif
  } else if (protocol == "globus") {
#ifdef CAE_ENABLE_GLOBUS
    HLOG(kDebug,
         "AssimilatorFactory: Creating GlobusFileAssimilator for 'globus' "
         "protocol");
    // For globus protocol, return a GlobusFileAssimilator
    return std::make_unique<GlobusFileAssimilator>(cte_client_);
#else
    // Globus support not compiled in
    HLOG(kError,
         "AssimilatorFactory: Globus protocol requested but Globus support not "
         "compiled in. "
         "Rebuild with -DCAE_ENABLE_GLOBUS=ON to enable Globus support.");
    return nullptr;
#endif
  }

  // Unsupported protocol
  HLOG(kError, "AssimilatorFactory: Unsupported protocol '{}'", protocol);
  return nullptr;
}

std::string AssimilatorFactory::GetUrlProtocol(const std::string& url) {
  // Check for standard URI format first (e.g., "globus://")
  size_t pos_standard = url.find("://");
  if (pos_standard != std::string::npos) {
    return url.substr(0, pos_standard);
  }

  // Fall back to custom format (e.g., "file::")
  size_t pos_custom = url.find("::");
  if (pos_custom != std::string::npos) {
    return url.substr(0, pos_custom);
  }

  return "";
}

}  // namespace wrp_cae::core
