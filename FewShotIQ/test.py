import os

directory = "pan_images/test/defective"

# Normalize the directory to use the OS-specific separator
normalized_directory = os.path.normpath(directory)

# Replace OS separators with dots for Python module paths
module_path = f"FewShotIQ.data.{normalized_directory.replace(os.sep, '.')}" if directory else "FewShotIQ.data"

print("Normalized directory:", normalized_directory)
print("Module path:", module_path)
