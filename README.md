***************
Summary

1) This folder contains the documents relevant for the understanding, processing, and visualisation for the dissertation pertaining to "Finding and predicting future values of patents" with dataset of patents approved from 1980 to 2010, with specific limitations, for the company Volkswagen AG. 
2) The purpose of this project is to do subclass level predictive analysis on the entire dataset from 2011 to 2015 for the company Volkswagen AG.

Volkswagen AG is a German automobile and tech company. 

According to WIPO:
B60 is the class for VEHICLES IN GENERAL
B60L is the subclass for the following: 

	1) PROPULSION OF ELECTRICALLY-PROPELLED VEHICLES
	2) SUPPLYING ELECTRIC POWER FOR AUXILIARY EQUIPMENT OF ELECTRICALLY-PROPELLED VEHICLES
	3) ELECTRODYNAMIC BRAKE SYSTEMS FOR VEHICLES IN GENERAL
	4) MAGNETIC SUSPENSION OR LEVITATION FOR VEHICLES; 
	    MONITORING OPERATING VARIABLES OF ELECTRICALLY-PROPELLED VEHICLES; 
            ELECTRIC SAFETY DEVICES FOR ELECTRICALLY-PROPELLED VEHICLES

With limitations found here: https://www.wipo.int/classifications/ipc/en/ITsupport/Version20170101/transformations/ipc/20170101/en/htm/B60.htm

***************
Descriptive Foldernames

1) The "Dissertation documents" folder contains relevant material pertaining to the assembly of the final document.
2) The "Images" folder contains relevant material pertaining to results' images and visualisation aids used in the final document. 
3) The "Data" folder contains relevant material pertaining to the original raw datasets, intermediary data, or external data used during the coding assembly.
4) The "Model Training" folder contains the training history plots of machine learning models used in this dissertation project.
4) "Main.ipynb" contains the entirety of the code used throughout this dissertation project.

***************
IMPORTANT

1) Main.ipynb is an Interactive Python Notebook (.ipynb) file designed to be a sufficiently standalone code, provided it has downloaded all the relevant modules. Use %pip install or relevant commands to import any additional modules in Main.ipynb should you require them prior to running.
2) You may opt to use Google Colab, an online platform by Google that allows anybody to write and execute arbitrary python code through the browser. Granted, this might take slightly longer to execute than running on your local coding environment.

***************
About raw datasets:
Raw data files in \Data are named after their respective levels: 

1) Lvl1 :- Section (e.g. A)
2) Lvl3 :- Class (e.g. A01)
3) Lvl4 :- Subclass (e.g. A01B)

For example, Lv3B601980To2010.txt pertains to the Class (Lv3) B60 patent data from 1980 to 2010. However, this numbering convention is different to literature, and only used for added clarity. 

In literature and in the final document, the following convention is used:

1) Lvl1 :- Section (e.g. A)
2) Lvl2 :- Class (e.g. A01)
3) Lvl3 :- Subclass (e.g. A01B)

Here, Lv3B601980To2010.txt would pertain to the Class (Lvl2) B60 patent data from 1980 to 2010. The only difference is in the naming convention of the files.

***************
Additional Information

1) \Data\training_columns.txt are subclasses *used* in model training after pre-processing. Subclasses not listed were not trained.

2) \Data\Models.txt contains model architectures of neural network models used for training. The labelling is consistent with the code and final document with two exceptions.
 - multi_conv_model in code is CNN in the final document
 - feedback_model in code is Autoregressive RNN in the final document 

3) \Data\VOW3.DE_daily.csv and \Data\VOW3.DE_monthly.csv contains daily and monthly data of stock prices for the VOW3DE stock from August 1998 to December 2010. The daily file starts from 22nd July 1998 and goes up to 2023.

***************
Builds and System Specifications
The code was run and implemented succesfully as of 31st August 2023 with the following specifications of Visual Studio Code by Microsoft:

1) Version: 1.81.1 (user setup)
2) Commit: 6c3e3dba23e8fadc360aed75ce363ba185c49794
3) Date: 2023-08-09T22:22:42.175Z
4) Electron: 22.3.18
5) ElectronBuildId: 22689846
6) Chromium: 108.0.5359.215
7) Node.js: 16.17.1
8) V8: 10.8.168.25-electron.0
9) OS: Windows_NT x64 10.0.22621

The code was run and implemented succesfully as of 31st August 2023 with the following Python extensions:

1) Python v2023.14.0 (Python 3.10.11)
2) Pylance v2023.8.50 
3) Jupyter v2023.7.1002162226

The code was run and implemented succesfully as of 31st August 2023 with the following system specifications:

1) OS Name: Microsoft Windows 11 Home
2) Processor : Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz, 2304 Mhz, 4 Core(s), 8 Logical Processor(s)

The code finished with no errors as of 31st August 2023 with the following runtime:

0.0 hour(s), 9.0 minutes, and 5.785089492797852 seconds.

***************
The information above is correct as of 3rd September 2023.
