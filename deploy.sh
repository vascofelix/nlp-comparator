#!/bin/bash
git pull
docker build -t nlp-comparator .
docker run -d --restart unless-stopped -p 5000:5000 -v ./default_docs:/app/default_docs --name nlp-comparator nlp-comparator
