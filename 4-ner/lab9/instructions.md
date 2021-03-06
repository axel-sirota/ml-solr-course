# Lab 9
## Named Entity Recognition with Solr

In this final notebook, we will enrich index time by adding tag attribute fields as default searcheable fields, to avoid indexing the whole description

0- Install en_core_web_trf

```
python3 -m spacy download en_core_web_trf
```

1- cd into the base of the repo
2- Run `jupyter notebook` from there (**not** the exercise folder of the lab!) and access the notebook in this lab, under `exercise`. Follow the instructions. Good luck!

Note: You can always check at the solutions!

3- Now that the dataset has been enriched with the tags, create a new core `airbnb_ner`

```
bin/solr create -c airbnb_ner
bin/solr config -c airbnb_ner -p 8983 -action set-user-property -property update.autoCreateFields -value false
```

4- Copy the schema we made in `dataset` folder and the solrconfig

```
cp ~/ml-solr-course/dataset/schema_ner.xml ~/solr-8.9.0/server/solr/airbnb_ner/conf/schema.xml
cp ~/ml-solr-course/dataset/solrconfig.xml ~/solr-8.9.0/server/solr/airbnb_ner/conf/solrconfig.xml
./solr-8.9.0/bin/solr restart
```

4'- Verify in Solr Admin that it loaded correctly

5- Use the POST tool to load the data into the airbnb_ner core

```
java -jar -Dc=airbnb_ner -Dauto ~/solr-8.9.0/example/exampledocs/post.jar ~/ml-solr-course/4-ner/lab9/expanded_dataset.csv
```

6- Search for any entity, like New York, and check it still returns it. If you check the schema.xml, we are not indexing the description or neighborhood

Amazing!! NER helped make this search engine more maintainable!
