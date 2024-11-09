from cat.mad_hatter.decorators import tool
from cat.looking_glass.stray_cat import StrayCat
from cat.log import log

from googlesearch import search
import trafilatura

from pydantic import BaseModel
import time

# for autocomplete
from langchain.docstore.document import Document
from cat.memory.vector_memory_collection import VectorMemoryCollection


class PageInfo(BaseModel):
    url: str
    title: str
    description: str
    content: str
    position: int

def get_page_content(url: str) -> str | None:
    """Returns the first 2000 characters of response or None if error occurred"""
    try:
        downloaded = trafilatura.fetch_url(url)
        return trafilatura.extract(downloaded)[:2000]
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return None

def search_engine_google(query: str, max_num_results=3) -> list[PageInfo]:
    """
    This tool uses Google's search engine to find information.
    """

    results: list[PageInfo] = []
    for result in search(query, num_results=10, advanced=True, safe=None):
        if len(results) == max_num_results:
            break

        page_content = get_page_content(result.url)
        if page_content is None:
            continue

        page_info = PageInfo(
            url=result.url,
            title=result.title,
            description=result.description,
            content=page_content,
            position=len(results)+1,
        )

        duplicate = False
        for semi_result in results:
            if semi_result.url == page_info.url:
                duplicate = True
                break

        if duplicate:
            continue

        results.append(page_info)

    return results

def store_documents_in_collection(
            collection_name: str,
            stray: StrayCat,
            docs: list[Document],
            source: str, # TODOV2: is this necessary?
            metadata: dict = {},
            show_logs: bool = True
        ) -> None:
        """Add documents to the Cat's declarative memory.

        This method loops a list of Langchain `Document` and adds some metadata. Namely, the source filename and the
        timestamp of insertion. Once done, the method notifies the client via Websocket connection.

        Parameters
        ----------
        docs : List[Document]
            List of Langchain `Document` to be inserted in the Cat's declarative memory.
        source : str
            Source name to be added as a metadata. It can be a file name or an URL.
        metadata : dict
            Metadata to be stored with each chunk.

        Notes
        -------
        At this point, it is possible to customize the Cat's behavior using the `before_rabbithole_insert_memory` hook
        to edit the memories before they are inserted in the vector database.

        See Also
        --------
        before_rabbithole_insert_memory
        """
        
        if show_logs:
            log.info(f"Preparing to memorize {len(docs)} vectors")

        # hook the docs before they are stored in the vector memory
        docs = stray.mad_hatter.execute_hook(
            "before_rabbithole_stores_documents", docs, cat=stray
        )

        # classic embed
        time_last_notification = time.time()
        time_interval = 10  # a notification every 10 secs
        stored_points = []
        for d, doc in enumerate(docs):
            if time.time() - time_last_notification > time_interval:
                time_last_notification = time.time()
                perc_read = int(d / len(docs) * 100)
                read_message = f"Read {perc_read}% of {source}"
                stray.send_ws_message(read_message)

                if show_logs:
                    log.warning(read_message)

            # add default metadata
            doc.metadata["source"] = source
            doc.metadata["when"] = time.time()
            # add custom metadata (sent via endpoint)
            for k,v in metadata.items():
                doc.metadata[k] = v

            doc: Document = stray.mad_hatter.execute_hook(
                "before_rabbithole_insert_memory", doc, cat=stray
            )
            inserting_info = f"{d + 1}/{len(docs)}):    {doc.page_content}"
            if doc.page_content != "":
                doc_embedding = stray.embedder.embed_documents([doc.page_content])

                try:
                    memory: VectorMemoryCollection = getattr(stray.memory.vectors, collection_name)
                except AttributeError:
                    log.error(f"Vector memory '{collection_name}' not found")
                    return

                stored_point = memory.add_point(
                    doc.page_content,
                    doc_embedding[0],
                    doc.metadata,
                )
                stored_points.append(stored_point)

                if show_logs:
                    log.info(f"Inserted into memory ({inserting_info})")
            else:
                if show_logs:
                    log.info(f"Skipped memory insertion of empty doc ({inserting_info})")

            # wait a little to avoid APIs rate limit errors
            time.sleep(0.05)

        # hook the points after they are stored in the vector memory
        stray.mad_hatter.execute_hook(
            "after_rabbithole_stored_documents", source, stored_points, cat=stray
        )

        # notify client
        finished_reading_message = (
            f"Finished reading {source}, I made {len(docs)} thoughts on it."
        )

        stray.send_ws_message(finished_reading_message)

        if show_logs:
            log.warning(f"Done uploading {source}")

def load_embedder_info(cat: StrayCat):
    embedder_size = len(cat.embedder.embed_query("hello world"))

    # Get embedder name (useful for for vectorstore aliases)
    if hasattr(cat.embedder, "model"):
        embedder_name = cat.embedder.model
    elif hasattr(cat.embedder, "repo_id"):
        embedder_name = cat.embedder.repo_id
    else:
        embedder_name = "default_embedder"

    return {
        "embedder_name": embedder_name,
        "embedder_size": embedder_size,
    }

def create_collection(cat: StrayCat, collection_name: str):
    """
    Create a new collection in cat

    Parameters
    ----------
    cat : StrayCat
        The Cat instance.
    memory_name : str
        The name of the new collections memory.
    """

    embedder_config = load_embedder_info(cat)

    collection = VectorMemoryCollection(
        client=cat.memory.vectors.vector_db,
        collection_name=collection_name,
        **embedder_config
    )

    cat.memory.vectors.collections[collection_name] = collection
    setattr(cat.memory.vectors, collection_name, collection)

def get_search_metadata(obj: PageInfo):
    """
    Extract metadata from a PageInfo object.

    Parameters
    ----------
    obj : PageInfo
        The PageInfo object from which to extract metadata.

    Returns
    -------
    dict
        The metadata extracted from the PageInfo object.
    """

    metadata = {
        "search": {
            "title": obj.title,
            "link": obj.url,
            "description": obj.description
        }
    }

    return metadata

def empty_collection(cat: StrayCat, collection_name: str):
    point_ids = []
    memory: VectorMemoryCollection = getattr(cat.memory.vectors, collection_name)
    for point in memory.get_all_points():
        point_ids.append(point.id)

    memory.delete_points(point_ids)

@tool(
    return_direct=True,
    examples=["Search information about", "Search on internet what is", "Search on internet", "Internet", "Search", "Search on internet informations"]
)
def search_with_google(query: str, cat: StrayCat):
    """
    What do you want to search internet?
    Your job is search in internet the query that user requests you to search.
    Your job is search in internet.

    Input is a valid searching query.
    """

    # load settings
    settings = cat.mad_hatter.get_plugin().load_settings()

    # max number of results
    max_num_search = settings["search_max_results"]

    # search
    cat.send_notification("Sto cercando in rete...")
    results = search_engine_google(query,  max_num_results=max_num_search)

    # create search collection
    search_collection_name = "search"

    if not hasattr(cat.memory.vectors, search_collection_name):
        create_collection(cat, search_collection_name)

    search_memory: VectorMemoryCollection = getattr(cat.memory.vectors, search_collection_name)

    # store results
    empty_collection(cat, search_collection_name)
    prompt_results = ""
    for info in results:
        prompt_results += f"Article Title: {info.title}\n"
        prompt_results += f"Content: {info.content}\n"

        store_documents_in_collection(
            collection_name=search_collection_name,
            stray=cat,
            docs=cat.rabbit_hole.string_to_docs(cat, info.content, source="testo"),
            source=info.url,
            metadata=get_search_metadata(info)
        )

    prompt = f"""
Rispondi alla DOMANDA dell'utente in modo chiaro, basati esclusivamente sulle informazioni
contenute nei RISULTATI nella ricerca.

DOMANDA:
{query}

RISULTATI:
{prompt_results}
"""
    
    response = cat.llm(prompt)

    response_chunks = cat.rabbit_hole.string_to_docs(cat, response, "testo")
    response_chunks_without_overlap = cat.rabbit_hole.string_to_docs(cat, response, "testo", chunk_overlap=0)

    new_response = ""
    for doc, chunk in zip(response_chunks, response_chunks_without_overlap):
        embedding = cat.embedder.embed_query(doc.page_content)

        new_response += chunk.page_content + "<br>"
        chunk_citations = set()
        chunk_citations_metadata = []
        for mem in search_memory.recall_memories_from_embedding(
            embedding, k=max_num_search
        ):
            mem_doc = mem[0]
            metadata = mem_doc.metadata["search"]

            if not metadata["link"] in chunk_citations:
                chunk_citations_metadata.append(metadata) 

            chunk_citations.add(metadata["link"])
        
        new_response += "\n".join([
            f"Citazione: <a href=\"{metadata['link']}\" target=\"_blank\">{metadata['title']}</a>"
            for metadata in chunk_citations_metadata
        ])

    new_response += "<br>Riferimenti:<br>" + "\n".join([
       f"<a href='{info.url}' target='_blank'>{info.title}</a>"
       for info in results
    ])


    return new_response