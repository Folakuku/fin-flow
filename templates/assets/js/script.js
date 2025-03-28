// script.js
// ... your existing script ...

// Example of ScrollReveal usage (add to your existing script)
ScrollReveal().reveal(".hero-section", { delay: 200 });
ScrollReveal().reveal(".video-section", { delay: 400 });
ScrollReveal().reveal(".input-section", { delay: 600 });
ScrollReveal().reveal(".pricing-section", { delay: 800 });
ScrollReveal().reveal(".gallery-section", { delay: 1000 });
ScrollReveal().reveal(".challenge-section", { delay: 1200 });
ScrollReveal().reveal(".community-section", { delay: 1400 });
ScrollReveal().reveal("#market-data", { delay: 1600 });
ScrollReveal().reveal("#finance-analytics", { delay: 1800 });
ScrollReveal().reveal("#finance-reports", { delay: 2000 });

document.getElementById("submit-btn").addEventListener("click", async () => {
  const stockSelect = document.getElementById("stock-select");
  const submitBtn = document.getElementById("submit-btn");
  const selectedStock = stockSelect.value;

  if (!selectedStock) {
    alert("Please select a stock to analyze.");
    return;
  }

  submitBtn.disabled = true;
  document.getElementById("loading").style.display = "block";

  try {
    const response = await fetch("/analyze-stock", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ stock_symbol: selectedStock }),
    });

    const data = await response.json();
    console.log("Analysis Response:", data);
  } catch (error) {
    console.error("Error analyzing stock:", error);
  } finally {
    document.getElementById("loading").style.display = "none";
    submitBtn.disabled = false;
  }
});
