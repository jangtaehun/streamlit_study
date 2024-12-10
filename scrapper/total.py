import requests
from bs4 import BeautifulSoup


class Total_scrapper:

    def __init__(self, term):
        self.term = term

    def total_berlin(self):
        url = f"https://berlinstartupjobs.com/skill-areas/{self.term}"
        response = requests.get(url, headers={"User-Agent": "Kimchi"})
        all_jobs = []
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            jobs = soup.find_all("li", class_="bjs-jlid")

            for job in jobs:
                company = job.find("a", class_="bjs-jlid__b").get_text()
                title = job.find("h4", class_="bjs-jlid__h").get_text()
                position = (
                    job.find("div", class_="links-box")
                    .get_text()
                    .replace("\n", "")
                    .replace("\t", " ")
                )
                link = job.find("h4", class_="bjs-jlid__h").find("a")["href"]
                job_data = {
                    "title": title,
                    "company": company,
                    "position": position.strip().replace(" ", ", "),
                    "link": link,
                }
                all_jobs.append(job_data)
        else:
            print("web3 is can't get jobs.")
        return all_jobs

    def total_web3(self):
        url = f"https://web3.career/{self.term}-jobs"
        request = requests.get(url, headers={"User-Agent": "Kimchi"})
        results = []

        if request.status_code == 200:
            soup = BeautifulSoup(request.text, "html.parser")
            for tr in soup.find_all("tr", class_="border-paid-table"):
                tr.decompose()
            jobs = soup.find_all("tr", class_="table_row")
            for job in jobs:
                table = job.find_all("td")[0]
                table_detail = table.find("div", class_="job-title-mobile")
                link = table_detail.find("a")["href"].strip()
                title = table_detail.get_text().strip()
                company = job.find_all("td")[1].get_text().strip()
                positions = job.find_all("td")[-1]
                total_position = positions.find_all("a")
                position_list = []
                for position_detail in total_position:
                    position_list.append(position_detail.string.strip())
                position = ", ".join(position_list)
                job_data = {
                    "title": title,
                    "company": company,
                    "position": position,
                    "link": f"https://remoteok.com{link}",
                }
                results.append(job_data)
        else:
            print("web3 is can't get jobs.")
        return results


class Result_scrapper:

    total_result = []

    def __init__(self, term):
        self.term = term

    def scrap(self):
        total_s = Total_scrapper(self.term)
        b = total_s.total_berlin()

        w3 = total_s.total_web3()
        total_result = b + w3
        return total_result
