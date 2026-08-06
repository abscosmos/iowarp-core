/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_API_H_
#define CLIO_CTE_API_H_

/**
 * Per-DLL export macro for clio_cte_core_client.
 *
 * Mirrors clio_runtime/api.h's CLIO_RUN_API. CMake sets
 * CLIO_CTE_CORE_BUILDING_DLL=1 PRIVATE on the clio_cte_core_client target;
 * consumers see the import form. On non-Windows platforms this expands to
 * nothing — visibility defaults already publish symbols. Needed because
 * WINDOWS_EXPORT_ALL_SYMBOLS only auto-exports functions; data globals
 * crossing DLL boundaries (g_cte_client and friends) still require
 * explicit dllexport/dllimport.
 */
#if defined(_WIN32)
#  ifdef CLIO_CTE_CORE_BUILDING_DLL
#    define CLIO_CTE_API __declspec(dllexport)
#  else
#    define CLIO_CTE_API __declspec(dllimport)
#  endif
#else
#  define CLIO_CTE_API
#endif

#include "clio_ctp/util/singleton.h"

/** Same as CTP_DEFINE_GLOBAL_PTR_VAR_{H,CC} but with CLIO_CTE_API decoration
 *  so the symbol is exported from clio_cte_core_client.dll and imported
 *  by every dependent module. */
#define CLIO_CTE_DEFINE_GLOBAL_PTR_VAR_H(T, NAME) \
  extern CLIO_CTE_API ::std::atomic<__TU(T) *> NAME;
#define CLIO_CTE_DEFINE_GLOBAL_PTR_VAR_CC(T, NAME) \
  CLIO_CTE_API ::std::atomic<__TU(T) *> NAME{nullptr};

#endif  // CLIO_CTE_API_H_
