""" Simple rule base chatbot with emotional monitoring """

from textblob import TextBlob
import twily_classifier as cl
import stop_words as stopwords
import json

with open('twilybot.json', 'r') as f:
    array = json.load(f)

CONVERSATION = array["conversations"]
BOT_NAME = 'Twily'
STOP_WORDS = stopwords.sw_list
neg_distribution = []


def sentiment(u_input):
    """Utilitarian function: Appends 'neg_distribution'
    with negative probability, returns Negative Probability"""

    blob_it = cl.trainer().prob_classify(u_input)
    npd = round(blob_it.prob("neg"), 2)
    neg_distribution.append(npd)
    return npd


def simplebot(user):
    """Rule base bot, takes an argument, user input in form of a string.
    In sequence will pre-process the string. Lower case, tokenize and remove
    stop words. iterates through CONVERSATION, if filtered_input intersects
    response_set is updated. if the set is empty, it returns a message,
    else it returns the longest string in the set"""

    user_input = user
    user_blob = TextBlob(user_input)

    lower_input = user_blob.lower()
    token_input = lower_input.words
    filtered_input = [w for w in token_input if w not in STOP_WORDS]

    response_set = set()
    for list in CONVERSATION:
        for sentence in list:
            sentence_split = sentence.split()
            if set(filtered_input).intersection(sentence_split):
                response_set.update(list)

    if not response_set:
        return "I am sorry, I don't have an answer, ask again"
    else:
        return max(response_set, key=len)


def escalation(uinput):
    """Monitors user sentiment index, takes an argument,
    user_input, in form of a string. If the emotional index,
    set by sentiment() and taken from neg_distribution,
    increases above a set threshold and it is sustained
    an automatic respose/action is triggered.
    simultaneously sending user_input to simplebot() for a
    response"""

    live_rep = f"Hi, I am Susan your live representative.\
 I see here {BOT_NAME} is unable to help you, what can I do for you?"

    sentiment(uinput)
    list_len = len(neg_distribution)
    bot_response = simplebot(uinput)
    if list_len > 3:
        last_3 = neg_distribution[-3:]
        if last_3[0] > .40 and last_3[0] <= last_3[1]:  # <= last_3[2]:
            return live_rep
        else:
            return bot_response
    else:
        return bot_response

if __name__ == '__main__':
    while True:
        try:
            user_input = input('You: ')
            print(escalation(user_input))
            print(neg_distribution)
        except (KeyboardInterrupt, EOFError, SystemExit):
            break
