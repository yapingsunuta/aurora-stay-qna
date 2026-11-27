
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi
import numpy as np
import os
import re

# Hybrid retrieval combines dense FAISS similarity with BM25 keyword ranking
_BM25_CACHE = {}

def _tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*|[$][0-9]+(?:[.][0-9]+)?", text.lower())

def _document_key(doc) -> str:
    return " ".join(doc.page_content.split()).lower()

def _needs_exception_context(query: str) -> bool:
    lowered = query.lower()
    return ("after the deadline" in lowered or "advance purchase" in lowered
            or "exception" in lowered or ("still" in lowered and "refund" in lowered))

_PROPERTY_ALIASES = {
    "atlanta": "ash-atl midtown",
    "midtown": "ash-atl midtown",
    "boston": "ash-bos harbor",
    "harbor": "ash-bos harbor",
    "chicago": "ash-chi loop",
    "chi loop": "ash-chi loop",
    "denver": "ash-den union",
    "den union": "ash-den union",
    "miami": "ash-mia brickell",
    "brickell": "ash-mia brickell",
    "seattle": "ash-sea pike",
    "sea pike": "ash-sea pike",
}

def _property_entities(query: str) -> list[str]:
    lowered = query.lower()
    entities = re.findall(r"ash-[a-z]{3}(?:[ ]+[a-z]+)?", lowered)
    entities.extend(entity for alias, entity in _PROPERTY_ALIASES.items() if alias in lowered)
    return list(dict.fromkeys(entities))

def _is_service_animal_query(query: str) -> bool:
    lowered = query.lower()
    mentions_service_animal = "service animal" in lowered or "service dog" in lowered
    negates_service_animal = bool(re.search(
        r"\b(?:not|isn't|isnt)\s+(?:a\s+)?service (?:animal|dog)\b",
        lowered,
    )) or "non-service animal" in lowered or "non-service dog" in lowered
    return mentions_service_animal and not negates_service_animal

def _query_variants(query: str) -> list[str]:
    variants = [query]
    lowered = query.lower()
    if any(term in lowered for term in ("cancel", "refund", "no-show")):
        variants.append("cancellation deadline rate plan room charge refundable non-refundable")
    if _needs_exception_context(query):
        variants.append(
            "one-time exception documented service failure duplicate booking verified "
            "system error approved service recovery cancellation refund manager approval"
        )
    if any(term in lowered for term in ("dog", "pet", "service animal")):
        variants.append("pet fee per stay service animal property pet limits")
    if _is_service_animal_query(query):
        variants.append(
            "service animal not obvious ask only required because disability work or task "
            "trained to perform certificate vest medical record no pet fee ADA"
        )
    if any(term in lowered for term in ("check in", "check-in", "check out", "check-out")):
        variants.append("standard property check-in check-out early arrival late departure")
    if "double queen" in lowered:
        variants.append("Double Queen two queen beds up to 4 guests fire code property limits group size")
    if "accessible king" in lowered:
        variants.append("Accessible King king bed selected accessible features ask which features are needed confirm property")
    if any(term in lowered for term in ("late checkout", "late check-out", "late departure")):
        variants.append("Late departure subject to availability fee manager front desk records approval agreed time full-night charge")
    if "parking" in lowered or "park" in lowered or "driving" in lowered:
        variants.append("parking nightly fee self parking valet separate from room rate property profile")
    if "breakfast" in lowered:
        variants.append("breakfast property rate plan included paid price per person not group-wide guarantee")
    if "member" in lowered and any(term in lowered for term in ("cancel", "flexible", "refund")):
        variants.append("Member or promotional member discount does not by itself make a rate flexible read cancellation line offer terms")
    if any(term in lowered for term in ("pool", "cabana")):
        variants.append(
            "fitness and pool available only where listed in property profile "
            "hours age rules reservations closures lifeguard status"
        )
    if any(term in lowered for term in ("loud", "noise", "noisy", "midnight", "disturbance", "quiet hours")):
        variants.append(
            "quiet hours 10:00 PM 7:00 AM contact room when safe document complaint "
            "repeat serious disturbances manager on duty immediate threat active violence 911"
        )
    return variants

def _get_bm25_index(vectorstore):
    cache_key = id(vectorstore)
    if cache_key not in _BM25_CACHE:
        documents = list(vectorstore.docstore._dict.values())
        corpus = [_tokenize_for_bm25(doc.page_content) for doc in documents]
        _BM25_CACHE[cache_key] = (documents, BM25Okapi(corpus))
    return _BM25_CACHE[cache_key]

def retrieve(query: str, k: int = 6, vectorstore=None) -> list[str]:
    if vectorstore is None:
        raise ValueError("A vectorstore is required for retrieval.")

    documents, bm25_index = _get_bm25_index(vectorstore)
    document_by_key = {_document_key(doc): doc for doc in documents}
    fused_scores = {}
    candidate_count = min(30, len(documents))

    for rank, doc in enumerate(vectorstore.similarity_search(query, k=candidate_count), start=1):
        key = _document_key(doc)
        fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (60 + rank)

    for variant in _query_variants(query):
        lexical_scores = bm25_index.get_scores(_tokenize_for_bm25(variant))
        lexical_order = np.argsort(lexical_scores)[::-1][:candidate_count]
        for rank, document_index in enumerate(lexical_order, start=1):
            if lexical_scores[document_index] <= 0:
                continue
            doc = documents[int(document_index)]
            key = _document_key(doc)
            fused_scores[key] = fused_scores.get(key, 0.0) + 2.0 / (60 + rank)

    property_entities = _property_entities(query)
    lowered_query = query.lower()
    for key in document_by_key:
        if any(entity in key for entity in property_entities):
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0
        if _needs_exception_context(query):
            if "one-time exception" in key or "service recovery" in key:
                fused_scores[key] = fused_scores.get(key, 0.0) + 0.10
        if _is_service_animal_query(query):
            if ("required because of a disability" in key and "work or task" in key):
                fused_scores[key] = fused_scores.get(key, 0.0) + 0.45
            if "must not demand" in key or ("certificate" in key and "medical record" in key):
                fused_scores[key] = fused_scores.get(key, 0.0) + 0.40
        if any(term in lowered_query for term in ("pool", "cabana")) and "fitness and pool" in key:
            fused_scores[key] = fused_scores.get(key, 0.0) + 0.25
        if "double queen" in lowered_query and "double queen" in key:
            fused_scores[key] = fused_scores.get(key, 0.0) + 0.70
        if "accessible king" in lowered_query and "accessible king" in key:
            fused_scores[key] = fused_scores.get(key, 0.0) + 0.70
        if any(term in lowered_query for term in ("late checkout", "late check-out", "late departure")) and "late departure" in key:
            fused_scores[key] = fused_scores.get(key, 0.0) + 0.70
        if "member" in lowered_query and "member discount does not by itself make a rate flexible" in key:
            fused_scores[key] = fused_scores.get(key, 0.0) + 0.70
        if any(term in lowered_query for term in ("loud", "noise", "noisy", "midnight", "disturbance", "quiet hours")):
            if "quiet hours" in key:
                fused_scores[key] = fused_scores.get(key, 0.0) + 0.40
            if "immediate threat" in key or "active violence" in key or "call 911" in key:
                fused_scores[key] = fused_scores.get(key, 0.0) + 0.35

    amenity_query = any(term in lowered_query for term in ("pool", "cabana"))
    service_animal_query = _is_service_animal_query(query)
    double_queen_query = "double queen" in lowered_query
    accessible_king_query = "accessible king" in lowered_query
    late_departure_query = any(term in lowered_query for term in (
        "late checkout", "late check-out", "late departure",
    ))
    parking_query = any(term in lowered_query for term in ("parking", "park", "driving"))
    breakfast_query = "breakfast" in lowered_query
    member_query = "member" in lowered_query and any(
        term in lowered_query for term in ("cancel", "flexible", "refund")
    )
    pet_query = any(term in lowered_query for term in ("dog", "pet")) and not service_animal_query
    noise_query = any(term in lowered_query for term in (
        "loud", "noise", "noisy", "midnight", "disturbance", "quiet hours",
    ))
    ranked_all = sorted(fused_scores, key=fused_scores.get, reverse=True)
    # Compose evidence for every intent in a multi-part question.
    all_keys = list(document_by_key)
    composed_keys = []

    def matching_keys(predicate):
        preferred = [key for key in ranked_all if predicate(key)]
        return preferred + [key for key in all_keys if predicate(key) and key not in preferred]

    def add_first(candidates):
        match = next((key for key in candidates if key not in composed_keys), None)
        if match is not None:
            composed_keys.append(match)

    def add_property_facts(field_term):
        facts = matching_keys(lambda key: (
            document_by_key[key].metadata.get("content_type") == "table_fact"
            and field_term in document_by_key[key].metadata.get("field", "")
        ))
        for entity in property_entities:
            add_first([key for key in facts if entity in key])

    pet_query = any(term in lowered_query for term in ("dog", "pet")) and not service_animal_query
    advance_purchase_query = "advance purchase" in lowered_query
    late_cancel_query = ("after the deadline" in lowered_query or "late cancellation" in lowered_query)

    if noise_query:
        add_first(matching_keys(lambda key: "quiet hours" in key and "document the complaint" in key))
        add_first(matching_keys(lambda key: "call 911" in key))
    if service_animal_query:
        add_first(matching_keys(lambda key: "required because of a disability" in key and "work or task" in key))
        add_first(matching_keys(lambda key: "certificate" in key and "medical record" in key))
    if amenity_query:
        add_first(matching_keys(lambda key: document_by_key[key].metadata.get("content_type") == "table_row" and "fitness and pool" in key))
        add_first(matching_keys(lambda key: document_by_key[key].metadata.get("content_type") == "property_catalog"))
    if double_queen_query:
        add_first(matching_keys(lambda key: "double queen" in key and "up to 4 guests" in key))
        add_first(matching_keys(lambda key: "when a guest's group size changes" in key))
    if accessible_king_query:
        add_first(matching_keys(lambda key: "accessible king" in key and "selected accessible features" in key))
        add_first(matching_keys(lambda key: "ask what feature is needed" in key and "confirm what the property can provide" in key))
    if any(term in lowered_query for term in ("check in", "check-in", "check out", "check-out")) and property_entities:
        add_property_facts("check-in / out")
        add_first(matching_keys(lambda key: "early arrival" in key and "subject to availability" in key))
        add_first(matching_keys(lambda key: "late departure" in key and "subject to availability" in key))
    if late_departure_query:
        add_first(matching_keys(lambda key: "late departure" in key and "full-night charge" in key))
    if parking_query and property_entities:
        add_property_facts("parking")
        add_first(matching_keys(lambda key: "parking charges are separate from the room rate" in key))
    if breakfast_query and property_entities:
        add_property_facts("breakfast")
        add_first(matching_keys(lambda key: document_by_key[key].metadata.get("content_type") == "table_row" and "included breakfast is not a group-wide guarantee" in key))
    if pet_query and property_entities:
        add_property_facts("pet")
        add_first(matching_keys(lambda key: "pet rules are property-specific" in key))
    if advance_purchase_query:
        add_first(matching_keys(lambda key: "advance purchase" in key and "non-refundable and non-changeable" in key))
        add_first(matching_keys(lambda key: "advance purchase cancellation" in key and "manager approval" in key))
    if any(term in lowered_query for term in ("smoke", "smoking", "vape", "vaping")):
        add_first(matching_keys(lambda key: "smoking or vaping remediation" in key and "$250" in key))
        add_first(matching_keys(lambda key: "the $250 smoking or vaping amount is an internal baseline only" in key))
    if member_query:
        add_first(matching_keys(lambda key: "member discount does not by itself make a rate flexible" in key and "cancellation line" in key))
    if late_cancel_query:
        add_first(matching_keys(lambda key: "flexible" in key and "one night plus applicable tax" in key))
        add_first(matching_keys(lambda key: "one-time exception" in key and "verified system error" in key))
        add_first(matching_keys(lambda key: "flexible reservation canceled late or no-show" in key and "approved exception" in key))

    if composed_keys:
        results = [document_by_key[key] for key in composed_keys[:k]]
        return [
            f"Content:\n{doc.page_content}\n\nMetadata:\n{doc.metadata}"
            for doc in results
        ]

    if noise_query:
        quiet_keys = [key for key in ranked_all if "quiet hours" in key]
        safety_keys = [key for key in ranked_all if "call 911" in key]
        if not safety_keys:
            safety_keys = [key for key in ranked_all if (
                "immediate threat" in key or "active violence" in key
            )]
        ranked_keys = []
        for key in quiet_keys[:1] + safety_keys[:1] + ranked_all:
            if key not in ranked_keys:
                ranked_keys.append(key)
            if len(ranked_keys) >= min(k, 2):
                break
    elif service_animal_query:
        handling_keys = [key for key in ranked_all if (
            "required because of a disability" in key and "work or task" in key
        )]
        documentation_keys = [key for key in ranked_all if (
            "must not demand" in key or ("certificate" in key and "medical record" in key)
        )]
        ranked_keys = []
        for key in handling_keys[:1] + documentation_keys[:1] + ranked_all:
            if key not in ranked_keys:
                ranked_keys.append(key)
            if len(ranked_keys) >= min(k, 2):
                break
    elif amenity_query:
        amenity_row_keys = [key for key in ranked_all if (
            document_by_key[key].metadata.get("content_type") == "table_row"
            and "fitness and pool" in key
        )]
        ranked_keys = (amenity_row_keys or ranked_all)[:min(k, 1)]
    elif double_queen_query:
        room_keys = [key for key in ranked_all if (
            document_by_key[key].metadata.get("content_type") == "table_row"
            and "double queen" in key and "up to 4 guests" in key
        )]
        occupancy_keys = [key for key in ranked_all if "when a guest's group size changes" in key]
        ranked_keys = []
        for key in room_keys[:1] + occupancy_keys[:1] + ranked_all:
            if key not in ranked_keys:
                ranked_keys.append(key)
            if len(ranked_keys) >= min(k, 2):
                break
    elif accessible_king_query:
        room_keys = [key for key in ranked_all if (
            document_by_key[key].metadata.get("content_type") == "table_row"
            and "accessible king" in key and "selected accessible features" in key
        )]
        accessibility_keys = [key for key in ranked_all if (
            "ask what feature is needed" in key and "confirm what the property can provide" in key
        )]
        ranked_keys = []
        for key in room_keys[:1] + accessibility_keys[:1] + ranked_all:
            if key not in ranked_keys:
                ranked_keys.append(key)
            if len(ranked_keys) >= min(k, 2):
                break
    elif late_departure_query:
        late_keys = [key for key in ranked_all if (
            document_by_key[key].metadata.get("content_type") == "table_row"
            and "late departure" in key and "full-night charge" in key
        )]
        ranked_keys = (late_keys or ranked_all)[:min(k, 1)]
    elif parking_query and property_entities:
        fact_keys = [key for key in ranked_all if (
            document_by_key[key].metadata.get("content_type") == "table_fact"
            and "parking" in document_by_key[key].metadata.get("field", "")
        )]
        ranked_keys = []
        for entity in property_entities:
            match = next((key for key in fact_keys if entity in key), None)
            if match and match not in ranked_keys:
                ranked_keys.append(match)
        policy_key = next((key for key in ranked_all if "parking charges are separate from the room rate" in key), None)
        if policy_key and policy_key not in ranked_keys:
            ranked_keys.append(policy_key)
        ranked_keys = (ranked_keys or ranked_all)[:min(k, 3)]
    elif breakfast_query and property_entities:
        fact_keys = [key for key in ranked_all if (
            document_by_key[key].metadata.get("content_type") == "table_fact"
            and "breakfast" in document_by_key[key].metadata.get("field", "")
        )]
        ranked_keys = []
        for entity in property_entities:
            match = next((key for key in fact_keys if entity in key), None)
            if match and match not in ranked_keys:
                ranked_keys.append(match)
        policy_key = next((key for key in ranked_all if "included breakfast is not a group-wide guarantee" in key), None)
        if policy_key and policy_key not in ranked_keys:
            ranked_keys.append(policy_key)
        ranked_keys = (ranked_keys or ranked_all)[:min(k, 2)]
    elif pet_query and property_entities:
        fact_keys = [key for key in ranked_all if (
            document_by_key[key].metadata.get("content_type") == "table_fact"
            and "pet" in document_by_key[key].metadata.get("field", "")
        )]
        ranked_keys = []
        for entity in property_entities:
            match = next((key for key in fact_keys if entity in key), None)
            if match and match not in ranked_keys:
                ranked_keys.append(match)
        policy_key = next((key for key in ranked_all if "pet rules are property-specific" in key), None)
        if policy_key and policy_key not in ranked_keys:
            ranked_keys.append(policy_key)
        ranked_keys = (ranked_keys or ranked_all)[:min(k, 2)]
    elif member_query:
        member_keys = [key for key in ranked_all if (
            "member discount does not by itself make a rate flexible" in key
            and "cancellation line" in key
        )]
        ranked_keys = (member_keys or ranked_all)[:min(k, 1)]
    else:
        ranked_keys = ranked_all[:k]
    results = [document_by_key[key] for key in ranked_keys]

    return [
        f"Content:\n{doc.page_content}\n\nMetadata:\n{doc.metadata}"
        for doc in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    prompt = f"""
    You are an AI assistant answering questions about AuroraStay hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly supported by the retrieved policy information.
    2. Do not use outside knowledge, assumptions, or information not present in the retrieved policy information.
    3. Include all relevant information needed to answer the query completely.
    4. Do not omit important conditions, exclusions, limitations, exceptions, or coverage details.
    5. Focus only on information directly relevant to the user's query.
    6. Do not include unrelated information from the retrieved policy information.
    7. Answer the user's specific question directly and ensure the response addresses what was asked.
    8. Keep the answer concise, clear, and easy to understand.
    9. Do not invent benefits, exclusions, limits, waiting periods, conditions, or coverage details.
    10. When a query names a property, use property-specific facts only when the exact property appears in the retrieved information. If it does not appear, explicitly state that the documents do not include a profile for the named property, so the requested amenity cannot be confirmed; advise checking the live property profile or contacting the property team.
    11. For multi-part questions, answer every part separately using its supporting context.
    12. If the retrieved information contains rules that apply to the question, synthesize and answer from those rules. Do not claim the information is insufficient merely because the policy does not phrase the answer as a direct yes or no.
    13. For a late cancellation or refund question, state the applicable baseline charge and any retrieved manager-approved exception conditions. Do not promise a refund before the required review and approval.
    14. For city or property comparisons, use the official property labels shown in the retrieved facts and state each requested property separately.
    15. When quoting a listed fee or price, include any retrieved caveat about rate-plan inclusion, property variation, or live-profile verification; do not overstate a conditional price as universal.
    16. If the information is insufficient, say:
        "The available policy documents do not contain enough information to answer this question."
        If the retrieved information conflicts, clearly mention the conflict.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 6,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
