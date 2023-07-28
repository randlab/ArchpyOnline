# AchPy Online

AchPy Online is a web interface for the ArchPy modeling code. It provides a convenient way to interact with the ArchPy library through a user-friendly web interface. The presented code is a feasability demonstration, using open data provided by the Swiss Topographic office (Swisstopo) and the Geneva Geological Survey. 

## Repository Contents

- **downloadfiles**: Files that are downloaded by the user using the app.
- **phenix**: Contains a slightly modified ArchPy package. Version 0.1 https://github.com/randlab/ArchPy
- **static**: Contains static files used in the web interface.
- **templates**: Contains HTML templates used for rendering web pages.
- **UpdateBoreholes.py**: Contains Python code for updating the local boreholes cache from the Canton of Geneva, and to preprocess them.
- **app.py**: Contains the main Python code for the web interface.
- **environment.yml**: YAML file specifying the environment dependencies, to be used with conda.
- **listpackages.txt**: text file specifying the environment dependencies, to be used with pip.

## Download Complete Test Data

In the `data` folder, you will find a download link to get complete test data. This data is essential for testing the functionality of the AchPy Online web interface thoroughly.

## Launching the Flask Server

To run the AchPy Online web interface, you need to launch a Flask server. Follow the steps below to get started:

1. First, make sure you have Python installed on your system.

2. Create a virtual environment (optional but recommended) to isolate the project dependencies:
```
python -m venv myenv
source myenv/bin/activate # On Windows: myenv\Scripts\activate
```

3. Install the required packages from the `environment.yml` file:

```pip install -r requirements.txt```

Geone should be installed manually from the geone repository : https://github.com/randlab/geone

4. Install and run Redis Server:
- Download and install Redis from the official website (https://redis.io/download) or use a package manager specific to your OS (e.g., apt, yum, brew).
- Start the Redis server on the default port (6379) by running `redis-server` in your terminal.

5. Launch Celery:
- Celery is used for background task processing in this web interface. Before running the app, make sure you have Celery installed in your virtual environment.
- Start Celery by running the following command in the project directory:
```celery -A app.celery worker --loglevel=info```

6. Finally, launch the Flask server:
```python flask app.py```

7. The web interface should now be accessible at `http://localhost:5000/` in your web browser.

8. Before trying the interface, make sure you have downloaded the necessary files in the `data` folder. See the note inside the folder. In addition run the command :
   ```
   python UpdateBoreholes.py
   ```
   in order to update the local cache of the boreholes database. 

## Languages

The project is primarily developed using the following languages:

- Python: 84.5%
- HTML: 11.8%
- CSS: 1.8%
- Jupyter Notebook: 1.4%
- JavaScript: 0.5%

For more details, please refer to the repository files and their respective contents.

Please ensure that you have completed all the setup steps, including installing Redis and running Celery, to ensure the proper functioning of the web interface. If you encounter any issues during setup or usage, consult the documentation or raise an issue on the repository page.



