# chemistry-grants-cluster
This script accepts National Science Foundation Awards data and sorts the awards into different topical clusters based on the award's abstract text.
The data I used focused on the Division of Chemistry and included abstracts from all awards made from 01/01/2015 to present. It can be downloaded here: 

https://www.nsf.gov/awardsearch/advancedSearchResult?PIId=&PIFirstName=&PILastName=&PIOrganization=&PIState=&PIZip=&PICountry=&ProgOrganization=03090000&ProgEleCode=&BooleanElement=All&ProgRefCode=&BooleanRef=All&Program=&ProgOfficer=&Keyword=&AwardNumberOperator=&AwardAmount=&AwardInstrument=&ActiveAwards=true&OriginalAwardDateOperator=After&OriginalAwardDateFrom=01%2F01%2F2015&StartDateOperator=&ExpDateOperator=

The script accepts the exported CSV version of this data.  The user provides a program element code, the number of topical clusters they want to investigate and the awards csv file. The abstract text is cleaned, lemmatized, and tokenized before it is used in an LDA model. The LDA model creates topic clusters. Each award can then be sorted into the topic areas.

The Script outputs a frequency plot showing how many awards are in each generated topic area, word clouds showing the words associated with each topic area (with the size signifiying the word's weight in each topic), and a t-SNE plot giving a visual representation of topic-awards clustering.

The core programs in the division of chemistry have the following program element codes:
6878 - Chemical Synthesis
6880 - Chemical Measurement and Imaging
6881 - Chemical Theory, Models and Computational Methods
6882 - Environmental Chemical Sciences
6883 - Chemistry of Life Processes
6884 - Chemical Catalysis
6885 - Macromolecular, Supramolecular and Nanochemistry 
9101 - Chemical Structure, Dynamics and Mechanisms A
9102 - Chemical Structure, Dynamics and Mechanisms B

Descriptions about each program and the science it covers can be found here : https://www.nsf.gov/funding/programs.jsp?org=CHE

This script could be used to explore other divisions as well, it just requires a little bit of digging in the exported data to establish the link between the program name and the program element code.
