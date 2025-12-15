from setuptools import setup, find_packages

setup(
    name="wdd3",          
    version="0.1.0",              
    author="Prajna Bhat",            
    author_email="prajnabhat111@gmail.com", 
    description="Video-based waggle dance detection and decoding developed as part of master thesis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},     
    install_requires=[

    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "wdd3_train = scripts.train:main",  
            "wdd3_infer = scripts.infer:main",
            "wdd3_visualise = scripts.visualise:main",
            "wdd3_preprocess = scripts.preprocess:main",
        ],
    },
)
