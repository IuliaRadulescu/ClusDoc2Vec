import mongoConnect
import requests
import xml.etree.ElementTree as ET
import urllib.request as libreq

mongoDb = mongoConnect.MongoDBConnector('ARXIVRobert6Clusters')

clusterTypes = [
	'cs.DB',
	'physics.geo-ph',
	'math.MG',
	'stats.ML',
	'q-bio.GN',
	'q-fin.EC'
]

k = 0

for start in range(0, 18000, 3000):

	with libreq.urlopen('http://export.arxiv.org/api/query?search_query=cat:'+ clusterTypes[k] +'&start=0&max_results=3000&min_results=3000') as url:
		r = url.read()

	root = ET.fromstring(r)

	for i, entry in enumerate(root.findall("{http://www.w3.org/2005/Atom}entry")):
		title = entry.find('{http://www.w3.org/2005/Atom}title').text
		summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
		author = entry.find('{http://www.w3.org/2005/Atom}author')

		authorName = author.find('{http://www.w3.org/2005/Atom}name').text

		published = entry.find('{http://www.w3.org/2005/Atom}published').text
		journal_ref = entry.find('{http://arxiv.org/schemas/atom}journal_ref')

		if(journal_ref != None):
			journal_ref = journal_ref.text
		
		category = entry.find('{http://arxiv.org/schemas/atom}primary_category')
		if (category != None):
			category = entry.find('{http://arxiv.org/schemas/atom}primary_category').get('term')
			document = {
				"title": title,
				"summary": summary,
				"author": authorName,
				"published": published,
				"journal_ref": journal_ref,
				"fetched_category": category,
				"category": clusterTypes[k]
			}
			mongoDb.insert('documents', document)
		else:
			print ('wrong: ' + str(i))

		print('final i = ' + str(i) + ' for k = ' + str(k))

	k = k + 1;


