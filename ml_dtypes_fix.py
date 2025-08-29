"""
Fix for ml_dtypes compatibility with ONNX 1.19.0
ONNX 1.19.0 expects these attributes but ml_dtypes may not have them
"""
import ml_dtypes
import numpy as np

# Create dummy float types using numpy.float16 as fallback for missing ONNX 1.19.0 attributes
missing_attrs = [
    'float4_e2m1fn',
    'float6_e2m3fn', 
    'float6_e3m2fn',
    'float8_e8m0fnu'
]

for attr in missing_attrs:
    if not hasattr(ml_dtypes, attr):
        # Use float8_e4m3fn if it exists, otherwise float16 as fallback
        fallback = getattr(ml_dtypes, 'float8_e4m3fn', np.float16)
        setattr(ml_dtypes, attr, fallback)
        print(f"Added missing ml_dtypes.{attr} -> {fallback}")

print("ml_dtypes compatibility fix for ONNX 1.19.0 applied")