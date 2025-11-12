@CLAUDE.md 

Use compiler agent.

Let's begin fixing the CMake errors. I'm currently getting an error, where we are
failing to find HermesShm. This is because we have added this as a subdirectory now,
so it is not installed before compiling. How should we fix this?

