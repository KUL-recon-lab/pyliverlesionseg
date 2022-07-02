import setuptools
import os

# read content of README.md
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="pyliverlesionseg",
    use_scm_version={'fallback_version':'unkown'},
    setup_requires=['setuptools_scm','setuptools_scm_git_archive'],
    author="Xikai Tang and Georg Schramm",
    author_email="xikai.tang@kuleuven.be, georg.schramm@kuleuven.be",
    description="CNN-based whole liver and liver lesion segmentation in CT and MR",
    long_description=long_description,
    license='Apache License 2.0',
    long_description_content_type="text/markdown",
    url="https://github.com/KUL-recon-lab/pyliverlesionseg",
    packages=setuptools.find_packages(exclude = ["data","scripts","test"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6, <3.7',
    install_requires=['h5py==2.10',
                      'tensorflow==1.12.0',
                      'keras==2.2.4',
                      'pymirc>=0.27',
                      'pynetdicom>=1.5'],
    entry_points = {'console_scripts' : ['pyliverlesionseg_predict=pyliverlesionseg.predict_liver_lesion_seg:main',
                                         'pyliverlesionseg_train=pyliverlesionseg.train_liver_lesion_seg:main',
                                         'pyliverlesionseg_liver_seg_service=pyliverlesionseg.dcm_server_liver_seg:main',
                                         'pyliverlesionseg_liver_lesion_seg_service=pyliverlesionseg.dcm_server_liver_lesion_seg:main']},
    include_package_data=True,
)
