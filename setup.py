from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="arsl-blip",
    version="1.0.0",
    author="Najiba Tagougui, Ansar Hani, Monji Kherallah",
    author_email="tag.najiba@gmail.com",
    description="Arabic Sign Language Recognition with BLIP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NajibaTagougui/ArSL-BLIP",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.5.1",
        "transformers>=4.36.0",
        "opencv-python>=4.9.0",
        "pillow>=10.1.0",
        "numpy>=1.24.3",
        "flask>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "arsl-predict=inference.predict:main",
            "arsl-demo=demo.app:main",
        ],
    },
)
