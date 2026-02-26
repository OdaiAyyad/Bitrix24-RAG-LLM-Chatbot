# Test script to verify all packages are installed correctly
# Save as: test_installation.py

print("Testing package imports...")
print("-" * 50)

try:
    import groq
    print("✓ groq imported successfully")
    print(f"  Version: {groq.__version__}")
except ImportError as e:
    print(f"✗ groq import failed: {e}")

try:
    from dotenv import load_dotenv
    print("✓ python-dotenv imported successfully")
except ImportError as e:
    print(f"✗ python-dotenv import failed: {e}")

try:
    import numpy as np
    print("✓ numpy imported successfully")
    print(f"  Version: {np.__version__}")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")

try:
    import pandas as pd
    print("✓ pandas imported successfully")
    print(f"  Version: {pd.__version__}")
except ImportError as e:
    print(f"✗ pandas import failed: {e}")

try:
    import faiss
    print("✓ faiss-cpu imported successfully")
except ImportError as e:
    print(f"✗ faiss-cpu import failed: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("✓ sentence-transformers imported successfully")
except ImportError as e:
    print(f"✗ sentence-transformers import failed: {e}")

print("-" * 50)
print("\nTesting environment variable...")

import os
api_key = os.getenv('GROQ_API_KEY')
if api_key:
    print(f"✓ GROQ_API_KEY is set (first 10 chars): {api_key[:10]}...")
else:
    print("✗ GROQ_API_KEY is not set")
    print("  Set it using: $env:GROQ_API_KEY='your-key-here'")

print("-" * 50)
print("\nAll basic checks complete!")
print("You can now run: python qa_routing_demo_with_llm.py")
