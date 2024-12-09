# storage_detection
Project to find objects in online auctions images and use chat to get information about specific items:
<img width="868" alt="image" src="https://github.com/user-attachments/assets/e43088a8-0cff-4474-97a2-03b32aeecdd5">


# 1. Web scraping:
1. Access website and get all auction links and auction IDs from the first page
![Screenshot 2024-12-08 211630](https://github.com/user-attachments/assets/3b235cf0-adbf-4974-b585-219ddec82c4c)

2. For all auctions:
   (open auction link; find all images; download images; add metadata to dictionary)
![Screenshot 2024-12-08 211815](https://github.com/user-attachments/assets/da823b3a-70fe-4409-9f76-3ef5ba2d8a01)

3. Navigate to the next page with next auctions and do steps 1-2
![Screenshot 2024-12-08 205849](https://github.com/user-attachments/assets/775cbfc5-4c3c-422b-aad0-8939d51e007d)

4. Save metadata (image names, image urls and IDs) into JSON
![Screenshot 2024-12-08 210133](https://github.com/user-attachments/assets/47874c85-0615-4658-906c-a021acea6297)

![Screenshot 2024-12-08 211157](https://github.com/user-attachments/assets/6d7243b7-e2a9-421c-b259-f616be1dbd7a)
![Screenshot 2024-12-01 201850](https://github.com/user-attachments/assets/43ae3902-b420-4c22-b85b-0516c8994091)


# 2. Neural network model:
    # Run object recognition on images and save recognazed objects with json data![Screenshot 2024-11-30 152300](https://github.com/user-attachments/assets/80121ec9-3c21-45c2-be17-0e6621ca76b5)
![Screenshot 2024-11-30 201030](https://github.com/user-attachments/assets/886d43e7-4bb7-42be-8732-7c73c4f803c0)

![Screenshot 2024-12-08 134716](https://github.com/user-attachments/assets/ece065f9-7d1f-4afd-bdc0-59b3543d5a51)



# 3. Chat through Telegram API:
    # Run ai agent with a tool (for future development) and run telegram bot
<img width="1655" alt="image" src="https://github.com/user-attachments/assets/afecc561-f905-437d-bea2-2ad310952eea">
<img width="1235" alt="image" src="https://github.com/user-attachments/assets/dae1af38-57a1-455e-9b3a-f0ab5f4da4f7">


