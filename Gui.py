import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pdfplumber
from datetime import datetime
import tempfile
import os

class TransactionAnalyzer:
    def __init__(self, pdf_path: str, language: str = 'en'):
        self.transactions = self._parse_transactions(pdf_path)
        self.language = language

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string with error handling."""
        try:
            # Clean and normalize the date string
            date_str = date_str.strip()
            # Adjust date format as per your input data
            return datetime.strptime(date_str, '%b %d, %Y')
        except ValueError as e:
            st.warning(f"Warning: Unable to parse date '{date_str}'.")
            return None

    def _parse_transactions(self, pdf_path: str) -> list:
        """Parse all transactions from the PDF file."""
        transactions = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text:
                    st.warning(f"Warning: Page {page_number + 1} is empty or could not be read.")
                    continue
                
                for line_number, line in enumerate(text.split('\n')):
                    try:
                        # Basic validation for line format (Adjust according to your data)
                        date_str = line[:12].strip()  # Assuming date is the first 12 characters
                        amount_index = line.rfind('₹')
                        if amount_index == -1:
                            continue  # Skip lines without an amount symbol

                        amount_str = line[amount_index + 1:].strip().replace(',', '')  # Remove commas in amounts
                        description = line[12:amount_index].strip()
                        transaction_type = 'CREDIT' if 'CREDIT' in line.upper() else 'DEBIT'
                        
                        # Parse amount and date
                        amount = float(amount_str)
                        date = self._parse_date(date_str)
                        if date is None:
                            continue  # Skip if date parsing fails

                        # Append valid transactions
                        transactions.append({
                            'date': date,
                            'description': description,
                            'type': transaction_type,
                            'amount': amount
                        })
                    except Exception as e:
                        st.warning(f"Warning: Failed to process line '{line}' on page {page_number + 1}.")
                        st.error(e)
                        continue

        # Return transactions sorted by date
        return sorted(transactions, key=lambda x: x['date'], reverse=True) if transactions else []

    def get_total_spending(self):
        return sum(t['amount'] for t in self.transactions if t['type'] == 'DEBIT')

    def get_total_income(self):
        return sum(t['amount'] for t in self.transactions if t['type'] == 'CREDIT')

    def get_balance(self):
        return self.get_total_income() - self.get_total_spending()

    def get_merchant_analysis(self) -> list:
        """Analyze spending by merchant."""
        from collections import defaultdict
        merchant_spending = defaultdict(float)
        for t in self.transactions:
            if t['type'] == 'DEBIT':
                merchant_spending[t['description']] += t['amount']
        return sorted(merchant_spending.items(), key=lambda x: x[1], reverse=True)

def create_charts(analyzer):
    df = pd.DataFrame([{
        'date': t['date'].strftime('%Y-%m-%d') if t['date'] else "Invalid Date",
        'amount': t['amount'],
        'type': t['type'],
        'description': t['description']
    } for t in analyzer.transactions])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Handle invalid dates gracefully

    # Daily spending chart
    daily_spending = df[df['type'] == 'DEBIT'].groupby('date')['amount'].sum().reset_index()
    fig_daily = px.line(daily_spending, x='date', y='amount',
                        title=translate("Daily Spending Trend", analyzer.language),
                        labels={'amount': translate("Amount", analyzer.language), 'date': translate("Date", analyzer.language)})
    
    # Top merchants chart
    merchant_data = pd.DataFrame(analyzer.get_merchant_analysis(), 
                                 columns=[translate('Merchant', analyzer.language), translate('Amount', analyzer.language)])
    fig_merchants = px.bar(merchant_data.head(10), x=translate('Merchant', analyzer.language), y=translate('Amount', analyzer.language),
                           title=translate("Top 10 Merchants by Spending", analyzer.language))
    fig_merchants.update_layout(xaxis_tickangle=-45)
    
    fig_pie = px.pie(df, values='amount', names='type',
                     title=translate("Transaction Distribution", analyzer.language))
    
    return fig_daily, fig_merchants, fig_pie

def translate(text, language):
    translations = {
        'Daily Spending Trend': {
            'en': 'Daily Spending Trend',
            'hi': 'दैनिक खर्च प्रवृत्ति',
            'mr': 'दैनंदिन खर्च प्रवृत्ती'
        },
        'Amount': {
            'en': 'Amount',
            'hi': 'राशि',
            'mr': 'रक्कम'
        },
        'Date': {
            'en': 'Date',
            'hi': 'तारीख',
            'mr': 'तारीख'
        },
        'Merchant': {
            'en': 'Merchant',
            'hi': 'व्यापारी',
            'mr': 'व्यापारी'
        },
        'Top 10 Merchants by Spending': {
            'en': 'Top 10 Merchants by Spending',
            'hi': 'खर्च के अनुसार शीर्ष 10 व्यापारी',
            'mr': 'खर्चानुसार शीर्ष १० व्यापारी'
        },
        'Transaction Distribution': {
            'en': 'Transaction Distribution',
            'hi': 'लेनदेन वितरण',
            'mr': 'व्यवहार वितरण'
        },
        'Total Spending': {
            'en': 'Total Spending',
            'hi': 'कुल खर्च',
            'mr': 'एकूण खर्च'
        },
        'Total Income': {
            'en': 'Total Income',
            'hi': 'कुल आय',
            'mr': 'एकूण उत्पन्न'
        },
        'Net Balance': {
            'en': 'Net Balance',
            'hi': 'कुल शेष',
            'mr': 'निव्वळ शिल्लक'
        },
        'Analysis Period': {
            'en': 'Analysis Period',
            'hi': 'विश्लेषण अवधि',
            'mr': 'विश्लेषण कालावधी'
        }
    }
    return translations.get(text, {}).get(language, text)

def main():
    st.set_page_config(page_title="Transaction Analyzer", layout="wide")
    st.title("📊 Transaction Analyzer")
    st.write("Upload your PDF statement to analyze transactions")

    col1, col2, col3 = st.columns(3)
    if col1.button('English'):
        selected_language = 'en'
    elif col2.button('हिन्दी'):
        selected_language = 'hi'
    elif col3.button('मराठी'):
        selected_language = 'mr'
    else:
        selected_language = 'en'

    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        analyzer = TransactionAnalyzer(pdf_path, language=selected_language)
        if not analyzer.transactions:
            st.warning("No transactions were extracted. Please check the PDF format.")
            return

        tab1, tab2, tab3 = st.tabs([
            translate("📈 Overview", selected_language), 
            translate("💰 Transactions", selected_language), 
            translate("📊 Charts", selected_language)
        ])

        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(translate("Total Spending", selected_language), f"{analyzer.get_total_spending():,.2f}")
            with col2:
                st.metric(translate("Total Income", selected_language), f"{analyzer.get_total_income():,.2f}")
            with col3:
                st.metric(translate("Net Balance", selected_language), f"{analyzer.get_balance():,.2f}")

            if analyzer.transactions:
                st.info(f"{translate('Analysis Period', selected_language)}: {analyzer.transactions[-1]['date'].strftime('%b %d')} to {analyzer.transactions[0]['date'].strftime('%b %d, %Y')}")

        with tab2:
            if analyzer.transactions:
                transactions_df = pd.DataFrame([{
                    translate('Date', selected_language): t['date'].strftime('%Y-%m-%d') if t['date'] else "Invalid Date",
                    translate('Description', selected_language): t['description'],
                    translate('Type', selected_language): t['type'],
                    translate('Amount', selected_language): f"{t['amount']:,.2f}"
                } for t in analyzer.transactions])

                st.dataframe(transactions_df, use_container_width=True)

                csv = transactions_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    translate("Download Transactions CSV", selected_language),
                    csv,
                    "transactions.csv",
                    "text/csv",
                    key='download-csv'
                )

        with tab3:
            if analyzer.transactions:
                fig_daily, fig_merchants, fig_pie = create_charts(analyzer)
                st.plotly_chart(fig_daily, use_container_width=True)
                st.plotly_chart(fig_merchants, use_container_width=True)
                col1, col2 = st.columns([2, 1])
                with col2:
                    st.plotly_chart(fig_pie, use_container_width=True)

        os.unlink(pdf_path)

if __name__ == "__main__":
    main()
