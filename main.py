import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

def main():
    information = """
    Ritesh Agarwal (born 16 November 1993) is an Indian billionaire entrepreneur[2] and the founder and CEO of PRISM, a hospitality chain.[3][4] He has been a panellist on Shark Tank India (season 3) in 2024.[5] Agarwal listed among Top 10 youngest Indian billionaires in Hurun Rich List of India 2024, with a net worth of $225 million (₹1,900 crore).[6][7]

Personal life
Ritesh Agarwal was born in a Marwari family[8] in Bissam Cuttack, Orissa, India and was brought up in Titilagarh.[9][10][11][12][13] His family operated a small shop in Rayagada, Odisha. He completed his schooling at Sacred Heart School, and later at St. John's Senior Secondary School, before moving to Delhi in 2011 for college.[14][15]

Agarwal married Geetansha Sood, a native of Lucknow, on 7 March 2023.[16][17]

Career
In 2011, Agarwal founded an Airbnb equivalent called "Oravel Stays".[18][19] Agarwal's enterprise was accelerated through the Venture Nursery program in 2012, and was later one of the winners of the 2013 Thiel Fellowship, receiving a US$100,000 grant.[20][21] The company launched as OYO Rooms in May 2013.[22][23][3]

By September 2018, the company had raised US$1 billion.[24] In July 2019, it was reported that Agarwal purchased US$2 billion in shares, tripling his stake in the company.[25][26][27]

He appeared on the Forbes 30 Under 30 list for Asia.[28]

In 2023, Agarwal became the youngest "Shark" to appear on Shark Tank India.[29]

In 2024, Agarwal continued to lead OYO Rooms as it built on its success from the previous year. The company focused on expanding its presence in key international markets, particularly in the Nordic countries, Southeast Asia, the United States, and the United Kingdom, where it had seen strong growth. Agarwal also made headlines for his prediction that spiritual tourism would become a significant driver of India’s tourism industry over the next five years. His insights were supported by data showing a 70% surge in bookings for holy destinations like Ayodhya on New Year’s Eve, surpassing popular tourist spots such as Goa and Nainital. [30]

In May 2024, OYO reported its first-ever profit after tax (PAT) of ₹229 crore for the financial year 2023–24, according to the company's annual report. This comes on the back of eight consecutive quarters of positive adjusted EBITDA, the company stated. OYO's adjusted EBITDA grew by 215% to reach ₹877 crore in FY24, up from ₹277 crore in FY23. [31] OYO's parent company, PRISM, secured shareholder approval for a ₹6,650 crore IPO in late December 2025, following strong Q1 FY26 financial results showing increased revenue and profit, with focus now on advancing listing preparations
    """

    prompt_template = """
    You are given with information {information} about a person. You need to provide below items.
    1. A short summary.
    2. Two interesting facts about them.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=prompt_template
    )

    llm = ChatOpenAI(
        model="gpt-5.4-mini",
        temperature=0
    )

    chain  = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
