from langchain.docstore.document import Document
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.utilities import (
    WikipediaAPIWrapper,
    GoogleSearchAPIWrapper,
)
from langchain.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain.chat_models import ChatOpenAI
import streamlit as st
import time
import os

# from langchain.document_loaders import UnstructuredURLLoader
# import pickle
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.callbacks import get_openai_callback
from qdrant_client import QdrantClient

# import pyperclip
from PyPDF2 import PdfReader
from constants import (
    # OPENAI_API_KEY,
    QDRANT_COLLECTION_ISLAMIC,
    QDRANT_API_KEY,
    QDRANT_HOST,
)
from utils import (
    count_words_with_bullet_points,
    create_word_docx,
)
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import io


def main_function():
    load_dotenv()
    keys_flag = False

    # with st.sidebar:
    #     st.subheader("Enter the required keys")

    #     st.write("Please enter your OPENAI API KEY")
    #     OPENAI_API_KEY = st.text_input(
    #         "OPENAI API KEY",
    #         type="password",
    #         value=st.session_state.OPENAI_API_KEY
    #         if "OPENAI_API_KEY" in st.session_state
    #         else "",
    #     )
    #     if OPENAI_API_KEY != "":
    #         keys_flag = True
    #         st.session_state.OPENAI_API_KEY = OPENAI_API_KEY
    keys_flag = True
    if keys_flag:  # or "OPENAI_API_KEY" in st.session_state:
        # os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        # search engines
        wiki = WikipediaAPIWrapper()
        wikiQuery = WikipediaQueryRun(api_wrapper=wiki)
        google = GoogleSearchAPIWrapper()
        duck = DuckDuckGoSearchRun()

        # Keyphrase extraction Agent
        llm_keywords = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo-16k")
        # keyword_extractor_tools = [
        #     Tool(
        #         name="Google Search",
        #         description="Useful when you want to get the keywords from Google about single topic.",
        #         func=google.run,
        #     ),
        #     Tool(
        #         name="DuckDuckGo Search Evaluation",
        #         description="Useful to evaluate the keywords of Google Search and add any missing keywords about specific topic.",
        #         func=duck.run,
        #     ),
        # ]
        # keyword_agent = initialize_agent(
        #     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        #     agent_name="Keyword extractor",
        #     agent_description="You are a helpful AI that helps the user to get the important keyword list in bullet points from the search results about specific topic.",
        #     llm=llm_keywords,
        #     tools=keyword_extractor_tools,
        #     verbose=True,
        #     handle_parsing_errors=True,
        # )
        keywords_prompt = """You are a helpful AI that helps the user to get the important keyword list in bullet points to write a reasearch paper about {topic} using the following information: {information}."""
        keywords_prompt_template = PromptTemplate(
            template=keywords_prompt,
            input_variables=["topic", "information"],
        )
        keywords_chain = LLMChain(
            llm=llm_keywords,
            prompt=keywords_prompt_template,
        )
        # title and subtitle Agent
        title_llm = ChatOpenAI(
            temperature=0.5, model="gpt-3.5-turbo-16k"
        )  # temperature=0.7
        # title_tools = [
        #     Tool(
        #         name="Intermediate Answer",
        #         description="Useful for when you need to get the title and subtitle for a research paper about specific topic.",
        #         func=google.run,
        #     ),
        # ]

        # title_agent = initialize_agent(
        #     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        #     agent_name="title and subtitle writer",
        #     agent_description="You are a helpful AI that helps the user to write a title and subtitle for a research paper about specific topic based on the given keywords",
        #     llm=title_llm,
        #     tools=title_tools,
        #     verbose=True,
        #     handle_parsing_errors=True,
        # )
        title_prompt = "You are a helpful AI that helps the user to write a title for a research paper about {topic} based on the following keywords {keywords} and using the following information {information}."
        title_prompt_template = PromptTemplate(
            template=title_prompt,
            input_variables=["topic", "keywords", "information"],
        )
        title_chain = LLMChain(
            llm=title_llm,
            prompt=title_prompt_template,
        )
        subtitle_prompt = "You are a helpful AI that helps the user to write a subtitle for a research paper about {topic} with a title {title} based on the following keywords {keywords} and using the following information {information}."
        subtitle_prompt_template = PromptTemplate(
            template=subtitle_prompt,
            input_variables=["topic", "title", "keywords", "information"],
        )
        subtitle_chain = LLMChain(
            llm=title_llm,
            prompt=subtitle_prompt_template,
        )
        # summarize the results separately
        summary_prompt = """Please Provide a summary of the following essay
        The essay is: {essay}.
        The summary is:"""
        summary_prompt_template = PromptTemplate(
            template=summary_prompt,
            input_variables=["essay"],
        )
        summarize_llm = ChatOpenAI(
            temperature=0, model="gpt-3.5-turbo-16k"
        )  # or OpenAI(temperature=0)
        summary_chain = LLMChain(
            llm=summarize_llm,
            prompt=summary_prompt_template,
        )
        # summarize the results together
        text_splitter = RecursiveCharacterTextSplitter(
            separators=[
                ".",
                "\n",
                "\t",
                "\r",
                "\f",
                "\v",
                "\0",
                "\x0b",
                "\x0c",
                "\n\n",
                "\n\n\n",
            ],
            chunk_size=1000,
            chunk_overlap=200,
        )
        summary_chain2 = load_summarize_chain(
            llm=summarize_llm,
            chain_type="map_reduce",
        )
        # create a summary agent
        summary_tools = [
            Tool(
                name="Intermediate Answer",
                func=wikiQuery.run,
                description="Use it when you want to get article summary from Wikipedia about specific topic",
            ),
            Tool(
                name="Google Search",
                description="Search engine useful when you want to get information from Google about single topic in general.",
                func=google.run,
            ),
            Tool(
                name="DuckDuckGo Search",
                description="Search engine useful when you want to get information from DuckDuckGo about single topic in general.",
                func=duck.run,
            ),
        ]
        summary_agent = initialize_agent(
            summary_tools,
            llm=summarize_llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            agent_name="Summary Agent",
            # verbose=True,
            handle_parsing_errors=True,
        )
        # create a research paper writer agent
        prompt_writer_outline = """You are an academic wrtier with expert writing skills and I want you to only write out the breakdown of each section of the research paper on the topic of {topic} 
        using the following information:
        keywords: {keywords}.
        The title is: {title}.
        The subtitle is: {subtitle}.
        relevant documents: {documents}.
        use the following template to write the research paper:
        [TITLE]
        [SUBTITLE]
        [introduction]
        [BODY IN DETIALED BULLET POINTS]
        [SUMMARY AND CONCLUSION]
        """
        prompt_writer = """You are an experienced academic researcher and academic writer using correct English grammar, where the quality would be suitable for academic journal publications.
                You will write an academic journal based on the {topic}. IT IS REALLY IMPORTANT THE RESEARCH PAPER MUST BE RELEVANT TO THE TOPIC.
                Follow this {outline} to to help with ideas and points around the topic of the journal.
                Don't use the same structure of the outline. Remove any bullet points and numbering systems.
        Source Of Information.
        The source of your information is the following documents: {documents}.
        The layout and structure of the output must be in long form sentences and long paragraph. Each paragraph must make its own point and must include reference to the documents.
        Write as a third party researcher, referring to the documents' author and their view is.
        Point of view
        You are not required to give your point of view. You must write from the perspective of a researcher providing the point of view of the author and refer to the author by name and quote their book using quotation marks and reference. It must be made clear who's opinion you are quoting.
        Structure Of Academic Journal
        The research paper should be structured implicitly, with an introduction at the beginning and a conclusion at the end of the research paper without using the words introduction, body and conclusion.
        You must write in long sentences, suitable for an academic style. Try to use different words and sentences to make the research paper more interesting.
        Use Keywords
        Check if the research paper contains these keywords {keywords} and if not, add them to the research paper.
        Word Count.
        Count the number of words in the research paper because the number of words must be maximized to be {wordCount} and add more words to the research paper to reach that number of words.
        """
        # prompt_writer = """You are an experienced writer and author and you will write a research paper in long form sentences using correct English grammar, where the quality would be suitable for academic publishing.
        #     First, Search about the best way to write a research paper about {topic}. THE RESEARCH PAPER MUST BE RELEVANT TO THE TOPIC.
        #     Second, use the following outline to write the research paper: {outline} because the research paper must write about the bullet points inside it and contain this information.
        #     Don't use the same structure of the outline.
        #     Remove any bullet points and numbering systems so that the flow of the research paper will be smooth.
        #     The research paper should be structured implicitly, with an introduction at the beginning and a conclusion at the end of the research paper without using the words introduction, body and conclusion.
        #     Try to use different words and sentences to make the research paper more interesting.
        #     The source of your information is the following documents: {documents}.
        #     Third, Check if the research paper contains these keywords {keywords} and if not, add them to the research paper.
        #     Fourth, Count the number of words in the research paper because the number of words must be maximized to be {wordCount} and add more words to the research paper to reach that number of words.
        #     Fifth, The research paper must be written in an academic style because it will be published as academic paper.
        #     """

        prompt_writer_template_outline = PromptTemplate(
            template=prompt_writer_outline,
            input_variables=[
                "topic",
                "title",
                "subtitle",
                "documents",
                "keywords",
            ],
        )

        prompt_writer_template = PromptTemplate(
            template=prompt_writer,
            input_variables=[
                "topic",
                "outline",
                "documents",
                "keywords",
                "wordCount",
            ],
        )

        # outline agent
        writer_outline_llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo-16k",
        )
        writer_chain_outline = LLMChain(
            llm=writer_outline_llm,
            prompt=prompt_writer_template_outline,
            # verbose=True,
        )
        # create a research paper writer agent
        writer_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
        writer_chain = LLMChain(
            llm=writer_llm,
            prompt=prompt_writer_template,
            # output_key="draft",
            # verbose=True,
        )

        reference_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")

        # evaluation agent
        evaluation_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")

        evaluation_prompt = """You are an expert academic papers editor and you will edit the draft to satisfy the following criteria:
        1- The research paper must be relevant to {topic}.
        2- The research paper must contain the following keywords: {keywords}.
        3- The research paper must contain at least {wordCount} words so use the summary {summary} to add an interesting senternces to the research paper.
        4- Sources will be used as references so at the end of each paragraph, you should add a reference to the source using the source number in []. 
        So, after each paragraph in the research paper, refer to the source index that most relevant to it using the source number in [].
        The used sources should be listed at the end of the research paper.
        the references should be in the following format: authors name, title of book, volume number, page number.
        5- The research paper must be written in an academic style because it will be published as academic paper.
        [Sources]
        {sources} 
        [DRAFT]
        {draft}
        The Result should be:
        1- All the mistakes according to the above criteria listed in bullet points:
        [MISTAKES]\n
        2- The edited draft of the research paper:
        [EDITED DRAFT]
        """
        evaluation_prompt_template = PromptTemplate(
            template=evaluation_prompt,
            input_variables=[
                "topic",
                "keywords",
                "wordCount",
                "summary",
                "draft",
                "sources",
            ],
        )

        evaluation_chain = LLMChain(
            llm=evaluation_llm,
            prompt=evaluation_prompt_template,
            # output_key="research paper",
            verbose=True,
        )

        # take the topic from the user
        #
        embeddings = OpenAIEmbeddings()

        client = QdrantClient(
            url=QDRANT_HOST,
            api_key=QDRANT_API_KEY,
            timeout=100000,
        )

        vectorStore = Qdrant(
            client=client,
            collection_name=QDRANT_COLLECTION_ISLAMIC,
            embeddings=embeddings,
        )
        retriever = vectorStore.as_retriever(search_kwargs={"k": 10})

        st.subheader(
            "This is a research paper writer agent that uses the following as sources of information:"
        )
        # unordered list
        st.markdown("""- Tafsir Al-Mizan for Quran""")

        myTopic = st.text_input("Write a research paper about: ", key="query")

        myWordCount = st.number_input(
            "Enter the word count of the research paper", min_value=100, max_value=3000, step=100
        )

        goBtn = st.button("**Go**", key="go", use_container_width=True)
        st.write("##### Current Progress")
        progress = 0
        progress_bar = st.progress(progress)
        keyword_list = ""
        title = ""
        subtitle = ""
        blog_outline = ""
        draft1 = ""
        draft1_reference = None
        draft2 = ""
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            [
                "Keywords list",
                "Title and Subtitle",
                "Blog Outline",
                "Draft 1",
                "Draft 2",
                "Final Blog",
            ]
        )
        if goBtn:
            try:
                with tab1:
                    with st.spinner("Generating the keywords list..."):
                        st.write("### Keywords list")
                        start = time.time()
                        # keyword_list = keyword_agent.run(
                        #     f"Search about {myTopic} and use the results to get the important keywords related to {myTopic} to help to write a research paper about {myTopic}."
                        # )
                        similar_docs = retriever.get_relevant_documents(
                            f"topic: {myTopic}"
                        )
                        keyword_list = keywords_chain.run(
                            topic=myTopic,
                            information=similar_docs,
                        )
                        end = time.time()
                        st.session_state.keywords_list_6 = keyword_list
                        # show the keywords list to the user
                        st.write(keyword_list)
                        st.write(
                            f"> Generating the keyword took ({round(end - start, 2)} s)"
                        )
                        progress += 0.16667
                        progress_bar.progress(progress)
                with tab2:
                    with st.spinner("Generating the title and subtitle..."):
                        # Getting Title and SubTitle
                        st.write("### Title")
                        start = time.time()
                        # title = title_agent.run(
                        #     f"Suggest a titel for a research paper about {myTopic} using the following keywords {keyword_list}?",
                        # )
                        # subtitle = title_agent.run(
                        #     f"Suggest a suitable subtitle for a research paper about {myTopic} for the a research paper with a title {title} using the following keywords {keyword_list}?",
                        # )
                        similar_docs = retriever.get_relevant_documents(
                            f"topic: {myTopic}, keywords: {keyword_list}"
                        )
                        title = title_chain.run(
                            topic=myTopic,
                            keywords=keyword_list,
                            information=similar_docs,
                        )
                        subtitle = subtitle_chain.run(
                            topic=myTopic,
                            title=title,
                            keywords=keyword_list,
                            information=similar_docs,
                        )
                        end = time.time()
                        st.session_state.title_6 = title
                        st.session_state.subtitle_6 = subtitle
                        st.write(title)
                        st.write("### Subtitle")
                        st.write(subtitle)
                        st.write(
                            f"> Generating the title and subtitle took ({round(end - start, 2)} s)"
                        )
                        progress += 0.16667
                        progress_bar.progress(progress)

                with tab3:
                    with st.spinner("Generating the research paper outline..."):
                        # write the research paper outline
                        st.write("### Blog Outline")
                        start = time.time()

                        print("reading vector store...")

                        print("Vector store created.")
                        similar_docs = retriever.get_relevant_documents(
                            f"title: {title}, subtitle: {subtitle}, keywords: {keyword_list}"
                        )
                        blog_outline = writer_chain_outline.run(
                            topic=myTopic,
                            title=title,
                            subtitle=subtitle,
                            documents=similar_docs,
                            keywords=keyword_list,
                        )
                        end = time.time()
                        st.session_state.blog_outline_6 = blog_outline
                        st.write(blog_outline)
                        # get the number of words in a string: split on whitespace and end of line characters
                        # blog_outline_word_count = count_words_with_bullet_points(blog_outline)
                        # st.write(f"> Blog Outline Word count: {blog_outline_word_count}")
                        st.write(
                            f"> Generating the first Blog Outline took ({round(end - start, 2)} s)"
                        )
                        progress += 0.16667
                        progress_bar.progress(progress)

                with tab4:
                    with st.spinner("Writing the draft 1..."):
                        # write the research paper
                        st.write("### Draft 1")
                        start = time.time()
                        similar_docs = retriever.get_relevant_documents(
                            f"research paper outline: {blog_outline}"
                        )
                        draft1 = writer_chain.run(
                            topic=myTopic,
                            outline=blog_outline,
                            documents=similar_docs,
                            keywords=keyword_list,
                            wordCount=myWordCount,
                        )
                        end = time.time()
                        st.session_state.draft1_6 = draft1
                        st.write(draft1)
                        # get the number of words in a string: split on whitespace and end of line characters
                        draft1_word_count = count_words_with_bullet_points(draft1)
                        st.write(f"> Draft 1 word count: {draft1_word_count}")
                        st.write(
                            f"> Generating the first draft took ({round(end - start, 2)} s)"
                        )

                        st.success("Draft 1 generated successfully")

                    with st.spinner("Referencing the first draft..."):
                        # reference the research paper
                        st.write("### Draft 1 References")
                        start = time.time()

                        # chain = RetrievalQAWithSourcesChain.from_llm(
                        #     reference_llm,
                        #     # chain_type="stuff",
                        #     retriever=retriever,
                        # )
                        chain = RetrievalQAWithSourcesChain.from_chain_type(
                            reference_llm,
                            chain_type="stuff",
                            retriever=retriever,
                        )

                        print("Chain created.")

                        draft1_reference = chain(
                            {
                                "question": f"First, Search for each paragraph in the following text {draft1} to get the most relevant source. \ Then, list those sources and order with respect to the order of using them in the research paper. The sources format should be: authors name, title of book, volume number, page number"
                            },
                            include_run_info=True,
                        )
                        # draft1_reference_from_chain_type = chain_from_chain_type(
                        #     {
                        #         "question": f"First, Search for each paragraph in the following text {draft1} to get the most relevant source. \ Then, list those sources and order with respect to the order of using them in the research paper. The sources should be the part of the document that contains the paragraph"
                        #     },
                        #     include_run_info=True,
                        # )
                        end = time.time()
                        st.session_state.draft1_reference_6 = draft1_reference
                        # draft1_reference = reference_agent.run(
                        #     f"First, Search for each paragraph in the following text {draft1} to get the most relevant links. \ Then, list those links and order with respect to the order of using them in the research paper."
                        # )
                        st.write("#### Relevant Text")
                        st.write(draft1_reference["answer"])
                        st.write("#### Relevant Sources")
                        st.write(draft1_reference["sources"])
                        st.write(
                            f"> Generating the first draft reference took ({round(end - start, 2)} s)"
                        )
                        progress += 0.16667
                        progress_bar.progress(progress)
                #########################################
                # evaluation agent
                # drafts = writer_evaluation_chain(
                #     {
                #         "topic": myTopic,
                #         "outline": blog_outline,
                #         "keywords": keyword_list,
                #         # "summary": tot_summary + tot_summary2,
                #         "wordCount": myWordCount,
                #     }
                # )
                # st.write("### Draft 1 V2")
                # st.write(drafts["draft"])
                # # get the number of words in a string: split on whitespace and end of line characters
                # draft1_word_count = count_words_with_bullet_points(drafts["draft"])
                # st.write(f"> Draft 1 word count: {draft1_word_count}")

                # st.write("### Draft 2")
                # st.write(drafts["research paper"])
                #######################################
                with tab5:
                    with st.spinner("Writing the second draft..."):
                        # edit the first draft
                        st.write("### Draft 2")
                        start = time.time()
                        draft2 = evaluation_chain.run(
                            topic=myTopic,
                            keywords=keyword_list,
                            wordCount=myWordCount,
                            summary=similar_docs,
                            draft=draft1,
                            sources=str(draft1_reference)
                            + str([doc.metadata for doc in similar_docs]),
                        )
                        end = time.time()
                        st.session_state.draft2_6 = draft2
                        st.write(draft2)
                        # get the number of words in a string: split on whitespace and end of line characters
                        draft2_word_count = count_words_with_bullet_points(draft2)
                        st.write(f"> Draft 2 word count: {draft2_word_count}")
                        st.write(
                            f"> Editing the first draft took ({round(end - start, 2)} s)"
                        )
                        st.success("Draft 2 generated successfully")
                        ########################################
                        progress += 0.16667
                        progress_bar.progress(progress)
                # edit the second draft
                with tab6:
                    with st.spinner("Writing the final research paper..."):
                        # write the research paper
                        st.write("### Final Blog")
                        start = time.time()
                        blog = evaluation_chain.run(
                            topic=myTopic,
                            keywords=keyword_list,
                            wordCount=myWordCount,
                            summary=similar_docs,
                            draft=draft2,
                            sources=str(draft1_reference)
                            + str([doc.metadata for doc in similar_docs]),
                        )
                        end = time.time()
                        st.session_state.blog_6 = blog
                        st.write(blog)
                        # get the number of words in a string: split on whitespace and end of line characters
                        blog_word_count = count_words_with_bullet_points(blog)
                        st.write(f"> Blog word count: {blog_word_count}")
                        st.write(
                            f"> Generating the research paper took ({round(end - start, 2)} s)"
                        )
                        st.success("Blog generated successfully")
                        progress = 1.0
                        progress_bar.progress(progress)
                        st.balloons()
                        doc = create_word_docx(myTopic, blog, None)
                        # Save the Word document to a BytesIO buffer
                        doc_buffer = io.BytesIO()
                        doc.save(doc_buffer)
                        doc_buffer.seek(0)
                        st.download_button(
                            label="Download Word Document",
                            data=doc_buffer.getvalue(),
                            file_name=f"{myTopic}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        )
                    # st.snow()
                # add copy button to copy the draft to the clipboard
                # copy_btn = st.button("Copy the research paper to clipboard", key="copy1")
                # if copy_btn:
                #     pyperclip.copy(draft1)
                # st.success("The research paper copied to clipboard")
            except Exception as e:
                st.error("Something went wrong, please try again")
                st.error(e)
        else:
            try:
                print("not pressed")
                with tab1:
                    if st.session_state["keywords_list_6"] is not None:
                        st.write("### Keywords list")
                        st.write(st.session_state["keywords_list_6"])
                        progress += 0.16667
                        progress_bar.progress(progress)

                with tab2:
                    if st.session_state["title_6"] is not None:
                        st.write("### Title")
                        st.write(st.session_state["title_6"])
                        st.write("### Subtitle")
                        st.write(st.session_state.subtitle_6)
                        progress += 0.16667
                        progress_bar.progress(progress)

                with tab3:
                    if st.session_state.blog_outline_6 is not None:
                        st.write("### Blog Outline")
                        st.write(st.session_state.blog_outline_6)
                        progress += 0.16667
                        progress_bar.progress(progress)

                with tab4:
                    if st.session_state.draft1_6 is not None:
                        st.write("### Draft 1")
                        st.write(st.session_state.draft1_6)
                        st.write("### Draft 1 References")
                        st.write(st.session_state.draft1_reference_6["answer"] + "\n\n")
                        st.write(st.session_state.draft1_reference_6["sources"])
                        progress += 0.16667
                        progress_bar.progress(progress)

                with tab5:
                    if st.session_state.draft2_6 is not None:
                        st.write("### Draft 2")
                        st.write(st.session_state.draft2_6)
                        progress += 0.16667
                        progress_bar.progress(progress)

                with tab6:
                    if st.session_state.blog_6 is not None:
                        st.write("### Final Blog")
                        st.write(st.session_state.blog_6)
                        # get the number of words in a string: split on whitespace and end of line characters
                        blog_word_count = count_words_with_bullet_points(
                            st.session_state.blog_6
                        )
                        st.write(f"> Blog word count: {blog_word_count}")
                        progress = 1.0
                        progress_bar.progress(progress)
                        st.success("Blog generated successfully")
                        st.balloons()
                        doc = create_word_docx(myTopic, st.session_state.blog_6, None)
                        # Save the Word document to a BytesIO buffer
                        doc_buffer = io.BytesIO()
                        doc.save(doc_buffer)
                        doc_buffer.seek(0)
                        st.download_button(
                            label="Download Word Document",
                            data=doc_buffer.getvalue(),
                            file_name=f"{myTopic}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        )
            except Exception as e:
                print(e)
    else:
        st.warning("Please enter your API KEY first", icon="⚠")


def main():
    st.set_page_config(page_title="Blog Writer Agent", page_icon="💬", layout="wide")
    st.title("Blog Writer Agent: Write an islamic research paper about any topic 💬")
    with open("./etc/secrets/config.yaml") as file:
        config = yaml.load(file, Loader=SafeLoader)
    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
        config["preauthorized"],
    )
    name, authentication_status, username = authenticator.login("Login", "main")
    if authentication_status:
        main_function()
        authenticator.logout("Logout", "main")
    elif authentication_status == False:
        st.error("Username/password is incorrect")
    elif authentication_status == None:
        st.warning("Please enter your username and password")


if __name__ == "__main__":
    with get_openai_callback() as cb:
        main()
        print(cb)
