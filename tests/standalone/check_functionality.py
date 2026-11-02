#!/usr/bin/env python
"""HippoRAG Project Functionality Check"""

print("=" * 60)
print("HippoRAG Project Functionality Check")
print("=" * 60)
print()

# Check 1: Import exceptions module
print("Check 1: Importing exceptions module...")
try:
    from hipporag.exceptions import HippoRAGException, LLMInitializationError

    print("  OK - Exceptions module imported successfully")
    print(f"  - Base exception: {HippoRAGException.__name__}")
    print(f"  - LLM exception: {LLMInitializationError.__name__}")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# Check 2: Import configuration module
print("Check 2: Importing configuration module...")
try:
    from hipporag.utils.env_config import EmbeddingConfig, HippoRAGConfig, LLMConfig

    print("  OK - Configuration module imported successfully")
    print(f"  - Main config: {HippoRAGConfig.__name__}")
    print(f"  - LLM config: {LLMConfig.__name__}")
    print(f"  - Embedding config: {EmbeddingConfig.__name__}")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# Check 3: Create configuration instance
print("Check 3: Creating configuration instance...")
try:
    config = HippoRAGConfig()
    print("  OK - Configuration instance created successfully")
    print(f"  - LLM Provider: {config.llm.provider}")
    print(f"  - Embedding Model: {config.embedding.model}")
    print(f"  - Vector Store Type: {config.vector_store.type}")
    print(f"  - Top K: {config.top_k}")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# Check 4: Test exception handling
print("Check 4: Testing exception handling...")
try:
    try:
        raise LLMInitializationError("Test error message")
    except HippoRAGException as e:
        print("  OK - Exception handling works correctly")
        print(f"  - Exception caught: {type(e).__name__}")
        print(f"  - Message: {str(e)}")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# Check 5: Verify test files
print("Check 5: Checking test files...")
try:
    import os

    test_files = []
    for root, dirs, files in os.walk("tests"):
        for file in files:
            if file.endswith(".py"):
                test_files.append(os.path.join(root, file))

    if test_files:
        print(f"  OK - Found {len(test_files)} test files")
        for f in sorted(test_files):
            print(f"  - {f}")
    else:
        print("  ERROR - No test files found")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# Check 6: Verify documentation
print("Check 6: Checking documentation files...")
try:
    import os

    doc_files = []
    for file in os.listdir("."):
        if file.endswith(".md") and any(
            x in file
            for x in [
                "REFACTORING",
                "QUICK",
                "IMPLEMENTATION",
                "FINAL",
                "DOCUMENTATION",
                "VERIFICATION",
                "CONCLUSIONS",
            ]
        ):
            doc_files.append(file)

    if doc_files:
        print(f"  OK - Found {len(doc_files)} documentation files")
        for f in sorted(doc_files):
            print(f"  - {f}")
    else:
        print("  ERROR - No documentation files found")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# Check 7: Verify configuration files
print("Check 7: Checking configuration files...")
try:
    import os

    config_files = ["pyproject.toml", ".pre-commit-config.yaml", ".bandit", ".env.example"]
    found = []
    for f in config_files:
        if os.path.exists(f):
            found.append(f)

    if found:
        print(f"  OK - Found {len(found)} configuration files")
        for f in found:
            print(f"  - {f}")
    else:
        print("  ERROR - No configuration files found")
except Exception as e:
    print(f"  ERROR: {e}")

print()
print("=" * 60)
print("PROJECT FUNCTIONALITY CHECK COMPLETE")
print("=" * 60)
