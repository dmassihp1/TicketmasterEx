# TicketmasterEx
Sentiment Analysis and Entity Recognition

The file contains code to classify the title column of the Ticketmaster.csv file into four sentiments (Need help understanding, Application Error, New feature request, and Terminate service), and to identify all the product entities.

The data was first pulled into a mySQL database table (‘ticketmaster9’), with two extra columns created for the output that will be produced: sentiment and product_entities. 

The first task is done using a pretrained RoBERTa model to create embeddings for each of the categories as well as the user comments (title column).  For each user comment, the cosine similarity (scale of 0 to 1) between that and all four sentiments is calculated, and it is assigned the sentiment with the highest similarity value.
The model was pretrained on Semantic Textual Similarity (STS) data from CommonCrawl and Wikipedia (from the Stanford Natural Language Inference Corpus), in which sentence pairs are labelled with entailment, contradiction, or neutral. I chose a model optimized for the STS task as it is the closest to the task we were trying to do here (assigning a 2-3 word phrase to a longer user comment), as opposed to question answering or paraphrase mining tasks for which other pretrained models were available.  I chose a RoBERTA model as it is an optimized version of BERT that is pretained on more data .  In contrast to BERT which involves a next sentence prediction layer (takes in two sentences at once and determines if the 2nd follows the 1st), the RoBERTa model uses only a mask token that is switched to various words during different training epochs.  In essence, the model uses an attention mechanism with multiple layers to capture the deep semantic relationships and dependencies between words, in order to produce the sentence vectors/embeddings.
The sentiments were then entered into the ticketmaster table in the sentiment column.
Given more time, I would also try assigning sentiments with the same process using some other powerful pretained transformer models, such as XLNet. I would also fine-tune the model by adding ground truth annotations to a random subset of the dataset and use a cosine similarity loss function: for example, for each user comment and sentiment pair, assigning a value between 0 or 1 based on how much it displays that sentiment. Additionally, according to the sentiments indicated by the model for each comment, I would also randomly select a subset of the test data to annotate in order to compute an evaluation metric (F1 score) to evaluate the model’s performance.

The second task was done using a pertained named entity recognition model from the flair package, which includes recognizing product entities. The following is the total set of entities recognized: Person, Norp, Fac, Org, Gpe, Loc, Product, Event, Work_Of_Art, Law, Language, Date, Time, Percent, Money, Quantity, Ordinal, Cardinal. The data consisted of telephone conversations, newswire, newsgroups, broadcast news, broadcast conversation, weblogs.  The model consisted of Flair and Glove embeddings fed into a bidirectional LSTM that learns the relationship between entities in surrounding words.  The entities outputted by the model were placed into the table in the product_entities column, where each row contained the original user comments with their entities embedded in the text  (after the corresponding word) using < >.

