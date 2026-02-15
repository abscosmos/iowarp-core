/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Chimaera Compose Utility
 *
 * Loads and processes a compose configuration to create pools
 * Assumes runtime is already initialized
 */

#include <iostream>
#include <string>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <chimaera/chimaera.h>
#include <chimaera/config_manager.h>
#include <chimaera/admin/admin_client.h>

void PrintUsage(const char* program_name) {
  HIPRINT("Usage: {} [--unregister] <compose_config.yaml>", program_name);
  HIPRINT("  Loads compose configuration and creates/destroys specified pools");
  HIPRINT("  --unregister: Destroy pools instead of creating them");
  HIPRINT("  Requires runtime to be already initialized");
}

int main(int argc, char** argv) {
  if (argc < 2) {
    PrintUsage(argv[0]);
    return 1;
  }

  bool unregister = false;
  std::string config_path;

  // Parse arguments
  int i = 1;
  while (i < argc) {
    std::string arg(argv[i]);
    if (arg == "--unregister") {
      unregister = true;
      ++i;
    } else {
      config_path = arg;
      ++i;
    }
  }

  if (config_path.empty()) {
    HLOG(kError, "Missing compose config path");
    PrintUsage(argv[0]);
    return 1;
  }

  // Initialize Chimaera client
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)) {
    HLOG(kError, "Failed to initialize Chimaera client");
    return 1;
  }

  // Load configuration
  auto* config_manager = CHI_CONFIG_MANAGER;
  if (!config_manager->LoadYaml(config_path)) {
    HLOG(kError, "Failed to load configuration from {}", config_path);
    return 1;
  }

  // Get compose configuration
  const auto& compose_config = config_manager->GetComposeConfig();
  if (compose_config.pools_.empty()) {
    HLOG(kError, "No compose section found in configuration");
    return 1;
  }

  HLOG(kInfo, "Found {} pools to {}",
       compose_config.pools_.size(), (unregister ? "destroy" : "create"));

  // Get admin client
  auto* admin_client = CHI_ADMIN;
  if (!admin_client) {
    HLOG(kError, "Failed to get admin client");
    return 1;
  }

  if (unregister) {
    // Unregister mode: destroy pools
    for (const auto& pool_config : compose_config.pools_) {
      HLOG(kInfo, "Destroying pool {} (module: {})",
           pool_config.pool_name_, pool_config.mod_name_);

      auto task = admin_client->AsyncDestroyPool(
          chi::PoolQuery::Dynamic(), pool_config.pool_id_);
      task.Wait();

      chi::u32 return_code = task->GetReturnCode();
      if (return_code != 0) {
        HLOG(kError, "Failed to destroy pool {}, return code: {}",
             pool_config.pool_name_, return_code);
        // Continue destroying other pools
      } else {
        HLOG(kSuccess, "Successfully destroyed pool {}", pool_config.pool_name_);
      }

      // Remove restart file if it exists
      namespace fs = std::filesystem;
      std::string restart_file = config_manager->GetConfDir() + "/restart/"
                                 + pool_config.pool_name_ + ".yaml";
      if (fs::exists(restart_file)) {
        fs::remove(restart_file);
        HLOG(kInfo, "Removed restart file: {}", restart_file);
      }
    }

    HLOG(kSuccess, "Unregister completed for {} pools",
         compose_config.pools_.size());
  } else {
    // Register mode: create pools
    for (const auto& pool_config : compose_config.pools_) {
      HLOG(kInfo, "Creating pool {} (module: {})",
           pool_config.pool_name_, pool_config.mod_name_);

      // Create pool asynchronously and wait
      auto task = admin_client->AsyncCompose(pool_config);
      task.Wait();

      // Check return code
      chi::u32 return_code = task->GetReturnCode();
      if (return_code != 0) {
        HLOG(kError, "Failed to create pool {} (module: {}), return code: {}",
             pool_config.pool_name_, pool_config.mod_name_, return_code);
        return 1;
      }

      HLOG(kSuccess, "Successfully created pool {}", pool_config.pool_name_);

      // Save restart config if restart_ flag is set
      if (pool_config.restart_) {
        namespace fs = std::filesystem;
        std::string restart_dir = config_manager->GetConfDir() + "/restart";
        fs::create_directories(restart_dir);
        std::string restart_file = restart_dir + "/" + pool_config.pool_name_ + ".yaml";

        // Write pool config wrapped in compose section so RestartContainers
        // can load it via ConfigManager::LoadYaml (which expects compose: [...])
        std::ofstream ofs(restart_file);
        if (ofs.is_open()) {
          // Indent the pool config under compose: list entry
          std::string indented;
          std::istringstream stream(pool_config.config_);
          std::string line;
          bool first = true;
          while (std::getline(stream, line)) {
            if (first) {
              indented += "  - " + line + "\n";
              first = false;
            } else {
              indented += "    " + line + "\n";
            }
          }
          ofs << "compose:\n" << indented;
          ofs.close();
          HLOG(kInfo, "Saved restart config: {}", restart_file);
        } else {
          HLOG(kWarning, "Failed to save restart config: {}", restart_file);
        }
      }
    }

    HLOG(kSuccess, "Compose processing completed successfully - all {} pools created",
         compose_config.pools_.size());
  }
  return 0;
}
