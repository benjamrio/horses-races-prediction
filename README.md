# Horse races prediction
Research project aiming to predict horse races results using state-of-the-art deep architectures.
First we investigate the elaboration of an *augmented music*, to provide an insightful time series describing horse historic performances. As of now, 3 features are collected for the performance of an horse at a certain race: 
- the result position
- the date of the race
- the cashprize for the first of the race
- 
## Installation
Clone the project.
From the root folder of the project, with an interpreter running python 3.9 :
```
$ virtualenv <env_name>
$ source <env_name>/bin/activate
(<env_name>)$ pip install -r requirements.txt
```
## Data
Data has been collected on different bookmakers websites, describing tens of thousands of races from 2016 to 2018. I will soon share it online.
## Project architecture
It is recommended to have input historic data in data > raw > 2016-2018_races > historic