# Use an official Python runtime as a parent image
FROM  continuumio/anaconda3

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN  pip install lxml
RUN  pip install prettytable
RUN  pip install requests
RUN  pip install tensorflow 
RUN  pip install imutils
RUN  pip install pandas
RUN  pip install seaborn
RUN  pip install matplotlib==2.2.2
RUN  pip install datetime
RUN  pip install numpy
RUN  pip install sklearn
RUN  pip install plotly
RUN  pip install urllib3



# Make port 80 available to the world outside this container
EXPOSE 443 8888 6006 80



# Run app.py when the container launches
CMD ["python", "trade_app3.py"]
