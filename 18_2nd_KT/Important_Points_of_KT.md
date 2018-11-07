# Important Points of KT

title: Important Points of KT  
categories: study in Unimelb

tags:  
- study  
- Unimelb  
- knowledge technology  
- review  



---

***Just for self-review, this document is not official or prefect. If there is any error, please tell me through the Issues in GitHub***

---

**Introduction to Knowledge Technology**

*Difference of data, information and knowledge*
>Data: measurements (bit patterns for computers)  
Information: processed data; patterns that are satisfied for given data  
Knowledge: information interpretted with respect to a user's context to extend human understanding in a given area (where we have data)

*Difinition of knowledge tasks and concrete tasks*
>Concrete tasks: mechanically processing data to an unambiguous solution; limited contribution to human understanding  
Knowledge tasks: data is unreliable or the outcome is ill-defined(usually both); computers mediate between the user an the data, where context (for the user) is critical; enhance human understanding.  

*Difinition of structured data, unstructured data and semi-structured data*
>Structured data: conforms to a schema, e.g. database  
Unstructured data: data without regular decomposable structure, e.g. plain text  
Semi-structured data: data which corresponds in part to a schema, but irregular or incomplete or rapidly changing; some important information is unavailable even with the schema.  
In practice, all data is semi-structured.

*Supervised learning*
>Classification: predicting a discrete class  
Regression: predicting a numeric quantity  

*Unsupervised learning*
>Association: detecting associations between features  
Information organisation; Clustering: grouping similar instances into clusters  
Reinforcement learning  
Recommender systems  
Anomaly/outlier detection  

*Some metacharacter in regular expressions*
>{ } [ ] ( ) ^ $ . | * + ? $ \

---
**Similarity and Probability**  

*TF-IDF*  

><img src="http://chart.googleapis.com/chart?cht=tx&chl=\small f_d" style="border:none;">, the number of terms contained in document d  
><img src="http://chart.googleapis.com/chart?cht=tx&chl=\small f_{d,t}" style="border:none;">, the frequency of term t in document d  
><img src="http://chart.googleapis.com/chart?cht=tx&chl=\small f_{ave}" style="border:none;">, the average number of terms contained in a document  
><img src="http://chart.googleapis.com/chart?cht=tx&chl=\small N" style="border:none;">, the number of documents in the collection  
><img src="http://chart.googleapis.com/chart?cht=tx&chl=\small f_t" style="border:none;">, the number of documents containing term t  
><img src="http://chart.googleapis.com/chart?cht=tx&chl=\small F_t" style="border:none;">, the total number of occurrences of t acroses all documents  
><img src="http://chart.googleapis.com/chart?cht=tx&chl=\small n" style="border:none;">, the number of indexed terms in the collection  
><img src="http://chart.googleapis.com/chart?cht=tx&chl=\small similarity=AB/|A||B|" style="border:none;">

*Entropy*
><img src="http://chart.googleapis.com/chart?cht=tx&chl=\small E=-p_i*log(p_i)" style="border:none;">

---
**Approximate Matching (not important)**

>Neighbourhood  
>Edit Distance  
>N-Gram Distance  
>Soundex  
>Accuracy
>Precision
>Recall

---
**Information Retrieval and Web Search**

*Definition of IR*
>IR is "the subfield of computer science that deals with storage and retreval of documents"

*the commonest mode of information seeking*
>Issue an initial query  
>Scan a list of suggested answers  
>Follow links to specific documents  
>Refine or modify the query  
>Use advanced querying features

*different types of "informational needs"*
>Requests for information, e.g. "global warming"  
>Factoid questions, e.g. "what is the melting point of lead"  
>Topic tracking, e.g. "what is the history of this news story"  
>Navigational, e.g. "University of Melbourne home page"  
>Service or transactional, e.g. "Mac powerbook"
>Geospatial, e.g. "Carlton restaurant"

*Definition of Relevance*
>A document is relevant if it contains knowledge that helps the user to resolve the information need

* One example of Boolean querying (do some questions)*
>diabets AND((NOT risk) OR juvenile) == 110 AND((NOT 011) OR 100) = 100

*features of Boolean querying*
>repeatable, auditable, and contollable.  
>no ranking and no control over result set size  
>difficult to incorporate useful heuristics

*Difinition of rank*
>The more similar or likely a document is, relative to the other documents in the collection, the higher its rank is.

*TF-IDF (do some questions)*
>same to similarity

*Evaluation Metrics*
$ J_\alpha(x) = \sum_{m=0}^\infty \frac{(-1)^m}{m! \Gamma (m + \alpha + 1)} {\left({ \frac{x}{2} }\right)}^{2m + \alpha} \text {，行内公式示例} $


Author: chuaan