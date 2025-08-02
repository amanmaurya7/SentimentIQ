export const sampleDatasets = {
  customerReviews: {
    name: "Customer Reviews Sample",
    description: "E-commerce product reviews with ratings and feedback",
    data: [
      { id: 1, text: "Great product! Fast delivery and excellent quality. Highly recommend!", category: "Product Quality", sentiment: "Positive" },
      { id: 2, text: "Terrible customer service. Had to wait 2 hours on hold just to speak to someone.", category: "Customer Service", sentiment: "Negative" },
      { id: 3, text: "Product arrived damaged. Very disappointed with the packaging.", category: "Delivery & Shipping", sentiment: "Negative" },
      { id: 4, text: "Amazing quality for the price. Will definitely buy again!", category: "Product Quality", sentiment: "Positive" },
      { id: 5, text: "Website is confusing and hard to navigate. Checkout process took forever.", category: "Technical Issues", sentiment: "Negative" },
      { id: 6, text: "Good product overall, but delivery took longer than expected.", category: "Delivery & Shipping", sentiment: "Neutral" },
      { id: 7, text: "Excellent customer support! They resolved my issue quickly and professionally.", category: "Customer Service", sentiment: "Positive" },
      { id: 8, text: "Product quality is decent but overpriced compared to competitors.", category: "Pricing", sentiment: "Neutral" },
      { id: 9, text: "Love the new features in the app! Much easier to use now.", category: "Technical Issues", sentiment: "Positive" },
      { id: 10, text: "Worst purchase I've ever made. Product broke after just one week.", category: "Product Quality", sentiment: "Negative" },
      { id: 11, text: "Fast shipping and good customer service. Product works as advertised.", category: "Customer Service", sentiment: "Positive" },
      { id: 12, text: "Too expensive for what you get. Not worth the money.", category: "Pricing", sentiment: "Negative" },
      { id: 13, text: "Pretty good product, no major complaints but nothing special either.", category: "Product Quality", sentiment: "Neutral" },
      { id: 14, text: "App keeps crashing on my phone. Very frustrating experience.", category: "Technical Issues", sentiment: "Negative" },
      { id: 15, text: "Outstanding quality and craftsmanship. Worth every penny!", category: "Product Quality", sentiment: "Positive" },
      { id: 16, text: "Delivery was on time and packaging was perfect. Great experience overall.", category: "Delivery & Shipping", sentiment: "Positive" },
      { id: 17, text: "Customer service was helpful but took a while to get a response.", category: "Customer Service", sentiment: "Neutral" },
      { id: 18, text: "Product didn't match the description on the website. Very misleading.", category: "Product Quality", sentiment: "Negative" },
      { id: 19, text: "Great value for money! Exceeded my expectations.", category: "Pricing", sentiment: "Positive" },
      { id: 20, text: "Website design is modern and easy to use. Shopping was a breeze.", category: "Technical Issues", sentiment: "Positive" }
    ]
  },
  
  socialMediaFeedback: {
    name: "Social Media Mentions",
    description: "Brand mentions and feedback from social media platforms",
    data: [
      { id: 1, text: "Just tried @brand's new service and I'm blown away! Incredible experience!", platform: "Twitter", sentiment: "Positive" },
      { id: 2, text: "Been using @brand for years and they keep getting better. Love the updates!", platform: "Facebook", sentiment: "Positive" },
      { id: 3, text: "@brand your app is broken again! Fix this please!", platform: "Twitter", sentiment: "Negative" },
      { id: 4, text: "Had a mixed experience with @brand. Some good, some bad.", platform: "Instagram", sentiment: "Neutral" },
      { id: 5, text: "Horrible customer service from @brand. Will never use again!", platform: "Facebook", sentiment: "Negative" },
      { id: 6, text: "@brand thanks for the quick response to my issue. Much appreciated!", platform: "Twitter", sentiment: "Positive" },
      { id: 7, text: "The new @brand update is okay I guess. Nothing groundbreaking.", platform: "Instagram", sentiment: "Neutral" },
      { id: 8, text: "@brand your prices are getting ridiculous. Time to find alternatives.", platform: "Twitter", sentiment: "Negative" },
      { id: 9, text: "Absolutely love @brand! Best decision I ever made.", platform: "Facebook", sentiment: "Positive" },
      { id: 10, text: "@brand has decent products but room for improvement in service.", platform: "Instagram", sentiment: "Neutral" }
    ]
  }
};

export const generateSampleCSV = (datasetKey: keyof typeof sampleDatasets): string => {
  const dataset = sampleDatasets[datasetKey];
  const headers = Object.keys(dataset.data[0]);
  
  const csvContent = [
    headers.join(','),
    ...dataset.data.map(row => 
      headers.map(header => `"${(row as any)[header]}"`).join(',')
    )
  ].join('\n');
  
  return csvContent;
};