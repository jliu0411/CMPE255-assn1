# Data Card

- **Dataset:** Online Retail
- **Source:** Kaggle mirror of the UCI transactional dataset
- **Unit:** One product line within an invoice
- **Key fields:** InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
- **Time coverage:** Original data covers approximately December 2010–December 2011
- **Bundled sample:** Synthetic, reproducible invoices with planted thematic affinities
- **Known issues:** Cancellations, returns, missing descriptions/customer IDs, duplicates, service lines, wholesale behavior, seasonality
- **Privacy:** CustomerID is not required for basket mining and is excluded from artifacts
- **Limitations:** Co-purchase does not demonstrate causal complementarity or customer intent
