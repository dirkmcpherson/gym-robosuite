from setuptools import setup, find_packages

setup(
    name='gym-robosuite',  # Replace 'your_package_name' with the name of your package
    version='0.1.0',          # Version number of your package
    author='',       # Your name or your organization's name
    author_email='dirkmcpherson@gmail.com',  # Your email or your organization's email
    description='Gym wrapper for robousite / mimicgen',  # A short description of your package
    long_description=open('README.md').read(),  # A long description from README.md
    long_description_content_type='text/markdown',  # Specifies that the long description is in Markdown
    url='http://github.com/dirkmcpherson/gym-robosuite',  # URL to your package's repository
    packages=find_packages(),  # Automatically find all packages and subpackages
    install_requires=[
        'numpy',   # List your package dependencies here
        'pandas'   # For example: numpy, pandas etc.
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',  # Choose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
        'Intended Audience :: Developers',  # Define who your audience is
        'License :: Apache 2.0 ',  # Choose the license
        'Programming Language :: Python :: 3',  # Specify the Python versions you support here. In particular, ensure
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.9',  # Specify which python versions you support
)

