import requests
from bs4 import BeautifulSoup
import re


def parse_and_save_images(base_url, start, end):
    for i in range(start, end + 1):
        url = base_url.format(i)
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            checkbox_input = soup.find('input', {'id': 'judged_checkbox'})
            if checkbox_input:
                data_label = checkbox_input['data-label']
                x_variable, y_variable = map(int,
                                             re.findall(r'X (\d+) Y (\d+)',
                                                        data_label)[0])

                view_image_trigger = soup.find('a',
                                               {'class': 'view_image_trigger'})
                if view_image_trigger:
                    competition_picture_guid = view_image_trigger[
                        'data-competition_picture_guid']

                    image_url = \
                        (f"https://www.botb.com/service/spottheball/viewimage?"
                         f"competitionPictureGuid={competition_picture_guid}"
                         f"&includeWinner=false"
                         f"&includeJudged=false"
                         f"&includeUserEntries=false"
                         f"&includeUserClosest=false")

                    image_response = requests.get(image_url)
                    if image_response.status_code == 200:
                        with open(f"images_training/{i}--{x_variable}-{y_variable}.png",
                                  'wb') as f:
                            f.write(image_response.content)
                            print(
                                f"Saved images_training/{i}--{x_variable}-{y_variable}.png")
                        with open(f"images_raw/{i}.png",
                              'wb') as f:
                            f.write(image_response.content)
                            print(
                                f"Saved images_raw/{i}.png")


if __name__ == "__main__":
    base_url = "https://www.botb.com/winners/sd{:03}"
    start_sequence = 3
    end_sequence = 86

    parse_and_save_images(base_url, start_sequence, end_sequence)
