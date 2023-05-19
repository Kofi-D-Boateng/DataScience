# 911 Capstone Project

# Project Abstract

This project focus is on analyzing 911 calls in specfic area.
The goal is to manipulate the data to help us understand how busy
this precient is, as well as the what type of calls they routinely
receive.

# Dataset

The data set used is a listing of 911 calls from Montgomery, Pennsylvania in 2016. The fields in the csv are as follows:

- Latitude: Physical Location
- Longitude: Physical Location
- Description: Metadata regarding the call
- Zipcode: the zipcode where the call was made in
- Title: Metadata regarding what the type of call
- Twp: Township (Location)
- Address: Location

# Data Insight

After cleaning the data up some, there is an average of 98001.2 data points within each columnm, with description being the outlier for least.

The next step was to create some more data points from the title. The typical title given to us followed the pattern of Type:Incident, delineated by a colon. I decided to use the Type of call as this could give us incite into the type of area Montgomery, PA is, and its demographics. [The graph given](./Reasons.png) reveals to us that EMS calls tend to be called in more, followed sequentially by traffic and fire. A guess based of this alone, suggest that the location is more likely an urban area, where the buildings frequently made from non-combustilbe materials more often, and the area also seems to be densily populated.

Some other graphs created below:

- [Reason Per Week](./ReasonsPerWeek.png)
- [Reason Per Month](./ReasonsPerMonth.png)
- [Call frequency throughout the year](./TimestampsOfEmergencies.png)
- [Townships vs Month](./TwpVsMonthLF.png)
- [EMS Calls](./EMSCalls.png)
- [Traffic Calls](./TrafficCalls.png)
- [Fire Calls](./FireCalls.png)

# Inferring on the data

After looking over the data points, and graphs created, there are a few things we can infer from them. - Calls for traffic peaking during the early part of the year, along with the max number of calls for EMS residing around the same time stamp, suggest that this area receives wintery weather conditions, and could potentially be a higher income area. Data points such as, but not limited to, median income, total car sales, number of dealerships, and unemployment rate in the area could allow for a better conclusion.

- The area received the worst part of winter early in the year of 2016, as the month following fall, did not receive as many calls for 911.

- The area of Montgomery, PA is a residental area without a college. This could be inferred by the fact that calls spiked during key vacation periods, such as winter and summer break, but fell during normal school periods. More demographic data would be needed to make this plausible.

# Conclusion

The dataset gave us good insight into the year of 2016 in the area of Montgomery, PA. There was a enough data to create meaningful graphs and infer thoughts regarding the area. I believe this dataset, combined with year-to-year datasets, would give us a better insight into the average 911 calls within the area, and could give also give us a superficial look into the demographics of the area.
