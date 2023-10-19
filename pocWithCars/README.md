how to run poc:

1. download car dataset (it is in one of the issues)
2. replace 01_prepare.2.py in dataset with the one in this dir
3. run the 01.prepare.2.py script
4. run the poc.py file with changed settings




Results analysis should consist of results to every epoch: 
--> test set: every class without TT class
--> val set: every class
--> test TT set: only TT class
