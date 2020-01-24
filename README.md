# senior-design


### Warning:
This repo is set to private so only people added as contributors can view. Even though it's private, please don't upload the data file to the repo just in case!

### Structure: 
The repo has a folder called ```data``` that contains the data spreadsheet. I've set up our ```.gitignore``` to ignore this file so that the data is not pushed to the repo if you are using GitHub desktop. If you use this structure, the notebooks should work. If not, they will error on the first cell and you will have to modify the file loading path to wherever you have the data stored on your laptop. The filepath should be ```../data/{filename}``` if your code is in the notebook or scripts subdirectories. 

### Resources:
1. How to use Git/GitHub: [https://guides.github.com/activities/hello-world/#commit]()
- [GitHub desktop](https://desktop.github.com/) can be super helpful to avoid having to use the terminal/your IDE's VCS.
1. [MapQuest Directions API](https://developer.mapquest.com/documentation/directions-api/): should be free, can be used to calculate driving distances in between warehouses