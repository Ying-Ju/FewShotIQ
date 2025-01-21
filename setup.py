from setuptools import setup, find_packages

setup(
    name="FewShotIQ",               # package name
    version="0.1.0",                # Initial version
    packages=find_packages(),       # Automatically find sub-packages
    install_requires=[],            # Add dependencies here, e.g., ["numpy", "torch"]
    include_package_data = True,
    package_data ={
        "FewShotIQ": ["data/**/*"]
    }, 
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
    extras_require={
        "clip": ["clip @ git+https://github.com/openai/CLIP.git"],  # Optional for CLIP
    },
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",        # Minimum Python version required
    description="FewShotIQ: Few-Shot Image Quality Inspection and Quantification",
    author="Fadel M Megahed, Ying-Ju Chen, Bianca Maria Colosimo, Marco Luigi Giuseppe Grasso, L. Allison Jones-Farmer, Sven Knoth, Hongyue Sun, Inez Zwetsloot",
    author_email="fmegahed@miamioh.edu, ychen4@udayton.edu, biancamaria.colosimo@polimi.it, marcoluigi.grasso@polimi.it, farmerl2@miamioh.edu,  knoth@hsu-hh.de, HongyueSun@uga.edu, i.m.zwetsloot@uva.nl",
    url="https://github.com/Ying-Ju/FewShotIQ",  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
