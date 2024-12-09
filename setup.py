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
    python_requires=">=3.7",        # Minimum Python version required
    description="FewShotIQ: Few-Shot Image Quality Inspection and Quantification",
    author="Fadel M Megahed, Ying-Ju Chen, L. Allison Jones-Farmer, Bianca Maria Colosimo, Hongyue Sun, Inez Zwetsloot, Marco Luigi Giuseppe Grasso, Sven Knoth",
    author_email="fmegahed@miamioh.edu, ychen4@udayton.edu, farmerl2@miamioh.edu, biancamaria.colosimo@polimi.it, HongyueSun@uga.edu, i.m.zwetsloot@uva.nl, marcoluigi.grasso@polimi.it, knoth@hsu-hh.de",
    url="https://github.com/yourusername/FewShotIQ",  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
