# Among Us 2021 Team

<img src="https://serpentine.ai/wp-content/uploads/2019/02/Final-design-serpentine.png" width="400px">

This is the repository of the Among Us team
from [E.S.A.I.V. Serpentine](https://www.serpentine.ai), and  is complementary to the paper titled [Automatically detecting player roles in Among Us](https://ieee-cog.org/2021/assets/papers/paper_249.pdf), which is published at the IEEE CoG 2021.
It contains the framework that was developed as part of the paper, as well as the data used in the paper. The framework was used to extract this data from videos with Among Us gameplay. 

## Framework

### setup:

For this project we used Pycharm for coding. But any python IDE will work as long as you can pip the libraries needed.

You will need the .pb model for the EAST text detection:

https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1

You will also need tesseract.
The (compiled) downloads for tesseract (windows/linux) can be found here:

https://digi.bib.uni-mannheim.de/tesseract/

5.0 was used, but any version 4 version should work as well (since no new 5.0 features are used)

### setting variables

After those two files are downloaded (and installed in the case of tessearct), you will need to set 3 variables in the framework.python


**Tesseract_location** :  path to the install location of tesseract. Make sure to include the / at the end as well

**model_detector** : Path to a .pb file contains trained detector network:


**video_location** : folder which contains all the videos you want to analyze. Make sure to include the / at the end as well

Once all those are set propperly, put the videos in your video location folder. When you run the framework.py it should now be analyzing the videos there.

## Among Us dataset
The folder "data" contains the chatlogs of 59 games of Among Us gameplay processed by the framework.

