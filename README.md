# Project TMA4212 - NumDiff


## How  to git for dummies:
Open terminal, and navigate to directory where you want to put this directory
type:

```
git clone https://github.com/jorgenriseth/tma4212-project.git
```

This will clone the repository into a directory called "tma4212-project". Start working as usual in this directory. When you feel like you have done a contribution that we should save,enter:

```
git status 
```

Your terminal will now display a list of files marked as modified, deleted, or untracked. They will probably be displayed in red. by writing:

```
git add [filename]
```
or
```
git add .
```

You will move the chosen files to the "staging area" (if  you use '.', you move all files to staging arear). These filenames will be colored green if running git status once more. If you are happy with the changes done to the files in the staging area, then these changes should be commited.

```
git commit -m "Short message explaining what you have done"
```

Now these changes are saved in your locally saved repository. It's possible to revert to this state at a later time, if you entirely fuck up ypur project. Before uploading these changes to our github rpository, you should first make sure you have merged your changes with those already in the repository. To do this:

```
git fetch
git pull
```
Now it's possible you will get some annoying merge changes and we will have to learn how to handle these properly. Finally 

```
git push
```

to upload your changes to our repository.
