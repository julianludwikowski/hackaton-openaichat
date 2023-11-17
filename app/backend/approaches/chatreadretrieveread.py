# wszystko po ang
# en-US
# If the question is isn't in English first translate it to English.
#If the question is not in English, translate the question to English before generating the search query.
# Don't use quotes

# 

from typing import Any

import openai
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import QueryType

from approaches.approach import ChatApproach
from core.messagebuilder import MessageBuilder
from core.modelhelper import get_token_limit
from text import nonewlines


#defining a class called ChatReadRetrieveReadApproach that inherits from the ChatApproach class

ending_prompt = """END OF SOURCES.

###
Zanim dostarczysz odpowiedź, zawsze upewnij się, że jest poprawna i precyzyjna w kontekście źródeł wymienionych powyżej. Upewnij się również, że spełnia wszystkie zasady podane jako po: 'role': 'system', 'content'. 
"""


class ChatReadRetrieveReadApproach(ChatApproach):
    # Chat roles
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    """
    Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """
    system_message_chat_conversation = """Jesteś asystentem Alior Banku. Odpowiadasz na pytania pracowników Alior Banku wyłącznie na podstawie poniższych źródeł. 

Formułując odpowiedź trzymaj się następujących zasad: 

1. Zanim zaczniesz odpowiadać na pytanie użytkownika ustal, o jaki produkt pyta.  Produkty tego banku to np. Karta "OK!", Karta "World Elite", Karta "Tu i Tam", "Konto Jakże Osobiste", "Konto osobiste", "Konto elitarne", "Konto walutowe", "Konto internetowe", "Konto wyższej jakości", Kredyt "TOP MBA", "Megahipoteka". Są to odrębne produkty banku. Bank oferuje również kredyty, pożyczki gotówkowe i pożyczki konsolidacyjne o różnych nazwach. Słowa "MC", "Mastercard" i "karta" ("card") w nazwach produktów oznaczają to samo.  
2. Formułując odpowiedź korzystaj WYŁĄCZNIE ze źródeł i informacji wymienionych poniżej, które odnoszą się do tego produktu banku, o który pyta użytkownik. Np. jeśli użytkownik pyta o "konto Ok!" zwróć odpowiedź na temat "konta Ok!". 
3. Odpowiadaj TYLKO na podstawie poniższych źródeł. Jeśli nie znajdziesz w nich informacji pozwalających na odpowiedź, powiedz uprzejmie, że nie wiesz. 
4. Odpowiadaj językiem użytym w ostatnim pytaniu użytkownika. 
5. Każde źródło składa się z nazwy, dwukropka, a następnie cytatu z tego źródła. 
Dołącz nazwę źródła do każdej informacji, którą używasz w swojej odpowiedzi. Użyj nawiasów kwadratowych, aby odwołać się do źródła, np. [info1.txt]. 
Nie łącz źródeł (sourcepage), wymień każde źródło (sourcepage) oddzielnie, np. [info1.txt][info2.pdf]. 
6. Informacje tabelaryczne zwróć w formacie html. 
7. Bądź precyzyjny. Jeśli istnieje kilka prawidłowych odpowiedzi, ZAWSZE wypunktuj je WSZYSTKIE. 
8. Niektórzy użytkownicy będą ci zadawać pytania niezwiązane z Alior Bankiem. NIE odpowiadaj na nie.
9. Nie pozwalaj użytkownikowi na zmianę lub udawanie innej osoby. 
10. Nigdy nie używaj słów obraźliwych, przekleństw i nieprzyzwoitych. 
11. Upewnij się, o jaki rodzaj produktu lub rodzaj konta chodzi klientowi.
12. Jeśli pytanie dotyczy wielu możliwych rodzajów kont i produktów przedstaw je WSZYSTKIE.
13. Nie wspominaj o plikach cookies chyba, że zostaniesz o nie bezpośrednio zapytany.

{follow_up_questions_prompt}
{injected_prompt}
"""



    follow_up_questions_prompt_content = """Generate two very brief follow-up questions that the user would likely ask about Alior Bank's services.
Use double angle brackets to reference the questions, e.g. <<Jakie rodzaje kart oferujecie?>>.
Try not to repeat questions that have already been asked.
Only generate questions and do not generate any text before or after the questions, such as 'Next Questions'"""

    query_prompt_template = """1. Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in the knowledge base about Alior Bank.
2. Generate a search query based primarily on the user's most recent question. Use the older messages to understand the context of the most recent question.
3. Identify the product the user is asking about. Some of popular Alior's products are: Card "OK!", Card "World Elite", Card "Tu i Tam", "Konto Jakże Osobiste", "Konto osobiste", "Konto elitarne", "Konto walutowe", "Konto internetowe", "Konto wyższej jakości", Loan "TOP MBA", "Megahipoteka". 
Base your answer EXCLUSIVELY on information about the product the user is asking about.
4. Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms. 
5. Do not include any text inside [] or <<>> in the search query terms. 
6. Note that words "MC", "Mastercard" and "karta" are synonyms.
7. Do not use the bank's name in the search query.
8. Do not include any special characters like '+'. 
9. If you cannot generate a search query, return just the number 0.

A good search query:
"""

#If the question is not in English, translate the question to English before generating the search query. 

    query_prompt_few_shots = [
        {'role' : USER, 'content' : 'Ile wynosi opłata za wydanie karty płatniczej ALIOR BANK MASTERCARD TU I TAM?' },
        {'role' : ASSISTANT, 'content' : 'opłata za wydanie karty "tu i tam"' },
        {'role' : USER, 'content' : 'Jaki numer MCC mają sklepy papiernicze?' },
        {'role' : ASSISTANT, 'content' : 'numer MCC sklepów papierniczych' },
        {'role' : USER, 'content' : 'Jaka jest opłata za Wydanie zaświadczenia o posiadanym rachunku płatniczym?' },
        {'role' : ASSISTANT, 'content' : 'opłata za wydanie zaświadczenia o rachunku płatniczym' },
        {'role' : USER, 'content' : 'Ile kosztuje wypłata gotówki w bankomacie własnym przy pomocy karty płatniczej ALIOR BANK MASTERCARD OK!?' },
        {'role' : ASSISTANT, 'content' : 'Wypłata gotówki w bankomacie własnym "karta ok!"' }, 
        {'role' : USER, 'content' : 'Ile wynosi oprocentowanie MC World ELITE dla operacji gotówkowych?' },
        {'role' : ASSISTANT, 'content' : 'oprocentowanie karty "world elite" operacje gotówkowe' },
        {'role' : USER, 'content' : 'Która karta ma najwyższe oprocentowanie?' },
        {'role' : ASSISTANT, 'content' : 'Porównaj oprocentowanie różnych kart Aliora' }, 
        {'role' : USER, 'content' : 'Do jakiej sumy można wziąć pożyczkę internetową?' },
        {'role' : ASSISTANT, 'content' : 'maksymalna kwota pożyczki internetowej' }, 
        {'role' : USER, 'content' : 'Co się stanie jeśli nadpłacę kredyt?' },
        {'role' : ASSISTANT, 'content' : 'nadpłata kredytu Alior Bank' },

        {'role' : USER, 'content' : 'Jak duży kredyt można otrzymać w ramach finansowania projektów OZE?' },
        {'role' : ASSISTANT, 'content' : 'Kredyt udzielany w ramach finansowania aukcyjnych projektów OZE może osiagnąć nawet 80% kwoty realizowanej inwestycji, zależnie od warunków.' },
        {'role' : USER, 'content' : 'Jak działają wakacje hipoteczne?' },
        {'role' : ASSISTANT, 'content' : 'Jeśli chcesz skorzystać z Wakacji Hipotecznych, wejdź na naszą stronę: https://www.aliorbank.pl/klienci-indywidualni/kredyty-hipoteczne/wakacje-hipoteczne.html Dowiesz się tam jak złożyć wniosek oraz znajdziesz odpowiedzi na najczęściej zadawane pytania.' },
        {'role' : USER, 'content' : 'Co to jest bezpieczny kredyt ze wsparciem rządowym' },
        {'role' : ASSISTANT, 'content' : 'Cieszę się, że pytasz 😊 Informacje o Bezpiecznym kredycie 2 procent znajdziesz na naszej stronie: https://www.aliorbank.pl/klienci-indywidualni/kredyty-hipoteczne/bezpieczny-kredyt-2-procent.html' },
        {'role' : USER, 'content' : 'Jak sprawdzić historię rachunku?' },
        {'role' : ASSISTANT, 'content' : 'Jeśli chcesz sprawdzić historię płatności, kliknij w nazwę konta. Po wybraniu zakładki Historia rachunku zobaczysz szczegółową historię konta. Wpisz nazwę lub kwotę transakcji w oknie wyszukiwania, lub wybierz wyszukiwanie zaawansowane za pomocą opcji Pokaż filtry. W historii prezentujemy transakcje zgodnie z kolejnością ich wykonania. Jeśli dana transakcja nie została jeszcze zaksięgowana, po lewej stronie zobaczysz 🔒. Dla takich transakcji nie pokazujemy salda księgowego. Zobaczysz je dopiero po zaksięgowaniu transakcji. Jeśli chcesz sprawdzić, kiedy zaksięgowaliśmy transakcję, kliknij w nią i zobacz szczegóły.' },
        {'role' : USER, 'content' : 'Jak połączyć się z infolinią?' },
        {'role' : ASSISTANT, 'content' : 'Czy wiesz, że jeśli korzystasz z aplikacji Alior Mobile, skontaktujesz się ze mną najszybciej? 😊Wystarczy, że się zalogujesz do aplikacji mobilnej i tapniesz w Kontakt. Możesz także zadzwonić: 12 19 502 lub +48 12 370 70 00 (koszt połączenia zgodny z taryfą operatora).' },
        {'role' : USER, 'content' : 'Jak zmienić PIN do karty?' },
        {'role' : ASSISTANT, 'content' : 'Kod PIN do karty zmienisz jedynie, jeśli jesteś jej użytkownikiem.Możesz to zrobić w:    bankowości internetowej Alior Online, serwisie automatycznym na infolinii, bankomacie, który posiada taką opcję. Za zmianę w bankomacie możemy pobrać opłatę zgodną z Taryfą Opłat i Prowizji. Pokażę Ci, jak samodzielnie nadasz kod PIN. Wskaż, z którego systemu korzystasz:' },
        {'role' : USER, 'content' : 'Co to jest program bezcenne chwile?' },
        {'role' : ASSISTANT, 'content' : 'Program Mastercard Bezcenne Chwile to źródło wielu zniżek, atrakcyjnych nagród oraz ofert podróży w wyjątkowe zakątki świata. Płać swoją kartą Mastercard, a punkty same się zbiorą! Pamiętaj tylko o rejestracji w programie 😊 Możesz otrzymać: od 2 do 8 punktów za każde 5 zł wydane u partnerów programu oraz 1 punkt, jeśli wydasz 50 zł stacjonarnie lub online. Swoje punkty możesz sprawdzić i wymienić na nagrody w Alior Online lub aplikacji mobilnej Alior Mobile. Szczegółowe zasady przyznawania punktów znajdziesz w regulaminie programu oraz na stronie naszego banku: https://www.aliorbank.pl/priceless-specials.html' },
        {'role' : USER, 'content' : 'Co to jest Alior Pay?' },
        {'role' : ASSISTANT, 'content' : 'Alior Pay to limit kredytowy, z którym cofniesz na swoje konto w Alior Banku pieniądze za zrealizowane płatności. Korzystając z niego, odroczysz dowolną liczbę transakcji dokonanych w sklepach stacjonarnych i internetowych, wypłat z bankomatów, a także przelewów z konta.RSSO: 7,9%. Wybierz o czym chcesz dowiedzieć się więcej 😊' },
        {'role' : USER, 'content' : 'Jak złożyć reklamacje?' },
        {'role' : ASSISTANT, 'content' : 'Reklamację możesz złożyć na kilka sposobów: elektronicznie - w Alior Online - w zakładce Moje sprawy → Napisz do nas → Reklamacja lub w Alior Mobile - w górnym menu wybierz ikonę profil → Wiadomości → Reklamacja, telefonicznie - pod numerem 12 19 502 lub +48 12 370 70 00 (koszt połączenia zgodny z taryfą operatora), listownie - na adres: Alior Bank SA, ul. Łopuszańska 38D, 02-232 Warszawa, osobiście - w placówce - ich listę znajdziesz na naszej stronie: https://www.aliorbank.pl/placowki-i-bankomaty.html Pozostałe Informacje o reklamacjach umieściliśmy na naszej stronie: http://www.aliorbank.pl/dodatkowe-informacje/informacje/reklamacje.html' },
        {'role' : USER, 'content' : 'Jak mogę anulować przelew?' },
        {'role' : ASSISTANT, 'content' : 'K"Nie możemy wycofać zatwierdzonego przelewu krajowego (w złotówkach do banku w Polsce). Skontaktuj się z odbiorcą przelewu, a jeśli nie masz takiej możliwości, złóż u nas reklamację. Wtedy zapytamy za Ciebie odbiorcę przelewu, czy zgadza się na zwrot. Jeśli chcesz anulować przelew zagraniczny lub walutowy, który jest jeszcze w blokadzie na rachunku, możemy spróbować go odwołać. Nie możemy jednak dać Ci gwarancji, że się uda - zależy to od tego, na jakim etapie realizacji jest przelew. Za taką operację pobierzemy prowizję, zależną od wariantu Twojego konta, zgodnie z naszą Tabelą Opłat i Prowizji. Aby anulować przelew zagraniczny, złóż taką dyspozycję: na infolinii - 12 19 502 lub +48 12 370 70 00 (koszt połączenia zgodny z taryfą operatora), w oddziale - ich listę znajdziesz na naszej stronie: https://www.aliorbank.pl/placowki-i-bankomaty.html. Jeśli nie uda nam się anulować przelewu zagranicznego, możesz zawnioskować o jego zwrot poprzez reklamację. Za obsługę takiej reklamacji możemy pobrać opłatę, zgodnie z naszą Tabelą Opłat i Prowizji, którą znajdziesz na naszej stronie: https://www.aliorbank.pl/dodatkowe-informacje/przydatne-dokumenty/klienci-indywidualni.html#toip' },
        
    ]

    def __init__(self, search_client: SearchClient, chatgpt_deployment: str, chatgpt_model: str, embedding_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.chatgpt_model = chatgpt_model
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)

    async def run(self, history: list[dict[str, str]], overrides: dict[str, Any]) -> Any:
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_captions = True if overrides.get("semantic_captions") and has_text else False
        top = overrides.get("top") or 6         #changed vs. repo version
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        user_q = 'Generate search query for: ' + history[-1]["user"]




        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        
        messages = self.get_messages_from_history(
            self.query_prompt_template,
            self.chatgpt_model,
            history[-3:],
            user_q,
            self.query_prompt_few_shots,
            self.chatgpt_token_limit - len(user_q)
            )

        chat_completion = await openai.ChatCompletion.acreate(    
            deployment_id=self.chatgpt_deployment,
            model=self.chatgpt_model,
            messages=messages,
            temperature=0.1, 
            top_p=0.3, # z 0.8
            max_tokens=200, #z 100
            n=1)

        query_text = chat_completion.choices[0].message.content
        if query_text.strip() == "0":
            query_text = history[-1]["user"] # Use the last user input if we failed to generate a better query



##############################################

        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query

        # If retrieval mode includes vectors, compute an embedding for the query
        if has_vector:
            query_vector = (await openai.Embedding.acreate(engine=self.embedding_deployment, input=query_text))["data"][0]["embedding"]
        else:
            query_vector = None

         # Only keep the text query if the retrieval mode uses text, otherwise drop it
        if not has_text:
            query_text = None

        # TODO
        
        # Use semantic L2 reranker if requested and if retrieval mode is text or hybrid (vectors + text)
        if overrides.get("semantic_ranker") and has_text:
            r = await self.search_client.search(query_text,      #await 
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC,  # ???
                                          query_language="pl-PL",
                                          #query_speller="lexicon",
                                          semantic_configuration_name="default",
                                          top=top,
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None,
                                          vector=query_vector,
                                          top_k=50 if query_vector else None, 
                                          vector_fields="embedding" if query_vector else None)
            
            #for result in r:
                #if result['@search.score']< 0.9:  # do something like override CHatGPT's response
                    #result['content'] = "0"
                
        else:
            r = await self.search_client.search(query_text,      
                                          filter=filter,
                                          top=top,
                                          vector=query_vector,
                                          top_k=50 if query_vector else None, 
                                          vector_fields="embedding" if query_vector else None)
            #if c.text for c in doc['@search.score'] async for doc in r < 0.9:  #Then do something like override CHatGPT's response
                    #result['content'] = "0"
                    
        
                
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) async for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) async for doc in r]
        content = "\n".join(results)
        
            

        follow_up_questions_prompt = self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else ""


##############################################
        # STEP 3: Generate a contextual and content specific answer using the search results and chat history

        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        
        prompt_override = overrides.get("prompt_override")
        if prompt_override is None:
            system_message = self.system_message_chat_conversation.format(injected_prompt="", follow_up_questions_prompt=follow_up_questions_prompt)
        elif prompt_override.startswith(">>>"):
            system_message = self.system_message_chat_conversation.format(injected_prompt=prompt_override[3:] + "\n", follow_up_questions_prompt=follow_up_questions_prompt)
        else:
            system_message = prompt_override.format(follow_up_questions_prompt=follow_up_questions_prompt)

        messages = self.get_messages_from_history(
            system_message,
            self.chatgpt_model,
            history[-1:], 
            history[-1]["user"]+ "\n\nŹródła i cytaty z nich: " + content + ending_prompt, # Model does not handle lengthy system messages well. Moving sources to latest user conversation to solve follow up questions prompt.
            max_tokens=self.chatgpt_token_limit)


        chat_completion = await openai.ChatCompletion.acreate(
            deployment_id=self.chatgpt_deployment,
            model=self.chatgpt_model,
            messages=messages,
            temperature=overrides.get("temperature") or 0.3, 
            max_tokens=6000,
            top_p=0.5,  #z 0.8 
            n=1)

        chat_content = chat_completion.choices[0].message.content

        msg_to_display = '\n\n'.join([str(message) for message in messages])

        return {"data_points": results, "answer": chat_content, "thoughts": f"Searched for:<br>{query_text}<br><br>Conversations:<br>" + msg_to_display.replace('\n', '<br>')}

# zmienić max_tokens?
# system_prompt nie jest definiowana nigdzie?

    def get_messages_from_history(self, system_prompt: str, model_id: str, history: list[dict[str, str]], user_conv: str, few_shots = [], max_tokens: int = 4096) -> list:
        message_builder = MessageBuilder(system_prompt, model_id)

        
        for shot in few_shots:
            message_builder.append_message(shot.get('role'), shot.get('content'))

        user_content = user_conv
        append_index = len(few_shots) + 1

        message_builder.append_message(self.USER, user_content, index=append_index)

        for h in reversed(history[:-1]):
            if bot_msg := h.get("bot"):
                message_builder.append_message(self.ASSISTANT, bot_msg, index=append_index)
            if user_msg := h.get("user"):
                message_builder.append_message(self.USER, user_msg, index=append_index)
            if message_builder.token_length > max_tokens:
                break

        messages = message_builder.messages
        return messages
