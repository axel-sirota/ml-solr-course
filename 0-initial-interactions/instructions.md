# Lab 0
##Interacting with PySolr

0- In the home folder of your Workstations you should have a Solr folder with Apache Solr. Start solr

```
cd solr-8.9.0
bin/solr start
```

1- Create an airbnb core and turn off managed schema

```
bin/solr create -c airbnb
bin/solr config -c airbnb -p 8983 -action set-user-property -property update.autoCreateFields -value false
```

2- Clone the repository (just in case) and copy the `schema.xml` and `solrconfig.xml` for this core
```
cd ~
rm -rf ~/ml-solr-course
git clone https://github.com/axel-sirota/ml-solr-course.git
cp ~/ml-solr-course/dataset/schema.xml ~/solr-8.9.0/server/solr/airbnb/conf/schema.xml
cp ~/ml-solr-course/dataset/solrconfig.xml ~/solr-8.9.0/server/solr/airbnb/conf/solrconfig.xml
./solr-8.9.0/bin/solr restart
```

2'- Go to Solr Admin page and verify the airbnb core loaded correctly

3- Use the POST tool to load the data into the airbnb core

```
java -jar -Dc=airbnb -Dauto ~/solr-8.9.0/example/exampledocs/post.jar ~/ml-solr-course/dataset/new_york_reduced.csv
```

4- Verify in Solr Admin (`http://localhost:8983`) the core has documents inserted

5- Run `jupyter notebook` and access the notebook in this lab, under `exercise`. Follow the instructions

Note: You can always check at the solutions!
