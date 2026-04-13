# Contributing

Thank you for interest in contributing to this parallel computing project!

## Code Style

### C++ Guidelines
- **Standard:** C++11 minimum
- **Indentation:** 4 spaces (no tabs)
- **Line length:** 100 characters max (readability)
- **Naming:**
  - Classes/Structs: `PascalCase`
  - Functions/Methods: `camelCase`
  - Constants: `UPPER_SNAKE_CASE`
  - Member variables: `snake_case_` (trailing underscore)

### Comments
- Use `//` for single-line comments
- Use `/* */` for multi-line comments
- Document all public functions with purpose, parameters, return values
- Explain non-obvious algorithms with inline comments

### CUDA Specifics
- Document kernel functions with block/thread configuration
- Always check CUDA error codes: `cudaGetLastError()`
- Memory transfers should be profiled (GPU ↔ Host overhead matters)

### MPI Specifics
- Document communication patterns (send/recv, collective operations)
- Include rank information in error messages
- Test with varying processor counts (2, 4, 8 if possible)

## Before Submitting

1. **Compile cleanly** — No warnings
2. **Run tests** — All tests pass
3. **Code review** — Review your own changes first
4. **Update docs** — Add/update relevant documentation
5. **Test edge cases** — Empty input, single element, power-of-2 vs non-power-of-2

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Test thoroughly
5. Commit with clear messages: `git commit -m "Add feature: description"`
6. Push to your fork
7. Open a Pull Request with:
   - Clear description of changes
   - Test results
   - Performance impact (if applicable)

## Testing

- Add unit tests for new functionality
- Ensure sequential and parallel versions produce same results
- Include performance comparisons for optimization changes

## Reporting Issues

- Describe the problem clearly
- Include environment (CUDA version, MPI implementation, OS)
- Provide minimal reproducible example if possible
- Share error messages and logs

## Areas for Contribution

- [ ] Documentation (algorithms, setup guides)
- [ ] Unit tests
- [ ] Performance benchmarking
- [ ] Error handling improvements
- [ ] Build system (CMake setup)
- [ ] Examples and sample data
- [ ] Bug fixes
- [ ] Code cleanup and refactoring

## Questions?

Open an issue with the `question` label or check existing documentation.

Thanks for helping make this project better!
