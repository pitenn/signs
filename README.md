pitenn/signs
============

As in description - project which is targeting a problem of traffic sign recognition using neural network.
----------------------------------------------------------------------------------------------------------
### Tips ###

You might want to use TensorFlow in version 1.5 if your CPU doesn't support AVX instruction set!
If so, download TensorFlow 1.5 in your venv. Otherwise your Python interpreter will get SIGKILL

### Resources ###

* Used photos for training are from [there](https://btsd.ethz.ch/shareddata/)
* Training set was [this](https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip)
* Testing set was [this](https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip)
*  This [link](https://drive.google.com/open?id=1W7cLXCiRc3f3CP4Y-u6s2YqYVaCIgVb8) 
leads to Google Drive directory within which you can find:
   *  model.json - Model architecture (already hardcoded in signs.py)
   *  model.h5 - Weights from taught neural network
*  Other useful links:
   *  [Tutorial](https://chsasank.github.io/keras-tutorial.html) - very good!
   *  [Other resources](https://medium.com/@waleedka/traffic-sign-recognition-with-tensorflow-629dffc391a6)

### Tutorial: Setting up development environment ###
In development we are using Python 3.x 

(@glaeqen 17 V 2018, I'm using 3.6.4)

Start:
1.  Download adequate package regarding the Python's virtualenv
    *  Your distro might using Python in version 2 or 3 as default
       E.g.: 
       *  pip -> Package manager for Python 2
       *  pip3 -> Package manager for Python 3

       or completely other way around
       *  pip2
       *  pip

       You get the idea. Be aware.
       
    *  When you got your 'pip' - download 'virtualenv'

       > pip install virtualenv

       Running pip _just like that_ installs packages into your system directories.
       Therefore you might need *SU privileges*
2.  Generate virtualenv directory
    *  Whole point of this mechanism is to have seperated hierarchy of Python's modules allowing us
       to deliver depenedencies _the easy way_.
    *  Command:

       > virtualenv <directory\_name>

       Where _<directory_name>_ is name where your venv will be stored. I usually use _venv_ as a name.
       
3.  Getting _into_ your virtual environment shell
    *  To use your freshly prepared virtual environment you have to source activation script into your shell
    *  Assuming that you are in the same directory as your venv directory you can achieve it by following command

       > source <virtualenv\_directory\_name>/bin/activate
       
       for bash shell. If you use different one, source different script, ask Google, do whatch' you gonna do.

    *  Now, you can notice name of your venv in parenthesis in the front of the prompt. If so, that means that
       you are inside your virtual environment, yey. E.g. you can use 'python' command to start interpreter -
       *<virtualenv\_directory\_name>/bin/python* one - which uses all dependencies from inside venv or use 'pip' to install
       new modules to venv!

4.  Install lacking dependencies
    *  Assuming there will be proper file in repository called *requirements.txt* which lists all dependencies
       required by project to run properly you can install them all at once

       > pip install -r requirements.txt

    *  Additionaly: File *requirements.txt* can be generated based on your current 'pip' package manager repository by:

       > pip freeze > requirements.txt

       Usually it will be performed by project manager so you don't have to bother.
    
And that's it. If you use virtualenv from inside of PyCharm, e.g. he will source activation script automatically 
if you use its internal (Alt + F12) terminal emulator. PyCharm strongly encourages and wraps around of concept 
of virtual environment so you don't get lost!
