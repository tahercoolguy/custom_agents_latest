from setuptools import setup, find_namespace_packages

setup(
    name="dynamic-agent-generator",
    version="0.1.0",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "smolagents>=1.2.2",
    ],
    python_requires=">=3.8",
    include_package_data=True,
) 