from setuptools import setup, find_packages

# Read the README.md file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="FewShotIQ",               # package name
    version="0.1.1",                # Initial version
    packages=find_packages(exclude=["tests", "tests.*"]),       # Automatically find sub-packages
    include_package_data = False,
    install_requires=[
        "openai>=0.27.0",
        "torch>=1.12.0",              # Adjust based on your PyTorch CUDA version
        "pandas>=1.1.0",
        "seaborn>=0.11.0",
        "numpy>=1.19.0",
        "requests>=2.25.0",
        "Pillow>=8.0.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=0.24.0",
        "textwrap3>=0.9.2",
    ],
    python_requires=">=3.10",        # Minimum Python version required
    description="FewShotIQ: Few-Shot Image Quality Inspection and Quantification",
    long_description=long_description,  # Include README.md as long_description
    long_description_content_type="text/markdown",  # Specify Markdown format
    author="Fadel M Megahed, Ying-Ju Chen, Bianca Maria Colosimo, Marco Luigi Giuseppe Grasso, L. Allison Jones-Farmer, Sven Knoth, Hongyue Sun, Inez Zwetsloot",
    author_email="fmegahed@miamioh.edu, ychen4@udayton.edu, biancamaria.colosimo@polimi.it, marcoluigi.grasso@polimi.it, farmerl2@miamioh.edu,  knoth@hsu-hh.de, HongyueSun@uga.edu, i.m.zwetsloot@uva.nl",
    url="https://github.com/Ying-Ju/FewShotIQ",  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
