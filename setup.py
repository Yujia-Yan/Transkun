import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# requirements= [i.strip() for i in open("requirements.txt").readlines()]

requirements = [
"ncls",
"pretty_midi",
"scipy",
"torchaudio",
"torch",
"mir_eval",
"pydub",
"seaborn",
"matplotlib",
"tensorboard",
"tqdm",
"torch_optimizer",
"sox",
"soxr"
]


setuptools.setup(
    name="transkun",
    version="2.0.0",
    author="Yujia Yan",
    description='Yet another tool for automatic piano transcription',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Yujia-Yan/Skipping-The-Frame-Level",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    package_data={'transkun.pretrained':['*.pt', '*.conf']},
    python_requires=">=3.6",
    entry_points={
        'console_scripts':[
            'transkun = transkun.transcribe:main',
            'transkunEval = transkun.computeMetrics:main'
        ]
    },
    install_requires=[
        requirements
    ],

)
