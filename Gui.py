import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pdfplumber
from datetime import datetime
import tempfile
import os

class TransactionAnalyzer:
    def __init__(self, pdf_path: str):
        self.transactions = self._parse_transactions(pdf_path)

    def _normalize_month(self, month: str) -> str:
        """Normalize month abbreviations."""
        month_map = {
            'Sept': 'Sep',
            'Sept.': 'Sep',
            'September': 'Sep',
            'Aug.': 'Aug',
            'August': 'Aug',
            'Oct.': 'Oct',
            'October': 'Oct',
            'Nov.': 'Nov',
            'November': 'Nov',
            'Dec.': 'Dec',
            'December': 'Dec',
            'Jan.': 'Jan',
            'January': 'Jan',
            'Feb.': 'Feb',
            'February': 'Feb',
            'Mar.': 'Mar',
            'March': 'Mar',
            'Apr.': 'Apr',
            'April': 'Apr',
            'May.': 'May',
            'June': 'Jun',
            'July': 'Jul'
        }
        return month_map.get(month, month)

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string with error handling."""
        try:
            # Clean and normalize the date string
            date_str = date_str.strip()
            month = date_str.split()[0]
            normalized_month = self._normalize_month(month)
            normalized_date = date_str.replace(month, normalized_month)
            
            # Try parsing with the standard format
            return datetime.strptime(normalized_date, '%b %d, %Y')
        except ValueError as e:
            st.error(f"Error parsing date: {date_str}")
            raise ValueError(f"Could not parse date: {date_str}") from e

    def _parse_transactions(self, pdf_path: str) -> list:
        """Parse all transactions from the PDF file."""
        transactions = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                for line in text.split('\n'):
                    try:
                        # Parse each transaction line
                        date_str = line[:12]  # Extract date
                        amount_index = line.rfind('₹')
                        if amount_index == -1:
                            continue  # Skip lines without an amount
                        amount_str = line[amount_index+1:].strip()  # Extract amount
                        transaction_type = 'CREDIT' if 'CREDIT' in line else 'DEBIT'
                        description = line[12:amount_index].strip()

                        # Convert amount to float, removing commas
                        amount = float(amount_str.replace(',', ''))

                        # Only add the necessary information to the transactions list
                        transactions.append({
                            'date': self._parse_date(date_str),
                            'description': description,
                            'type': transaction_type,
                            'amount': amount
                        })
                    except Exception as e:
                        st.error(f"Error processing line: {line}")
                        st.exception(e)
                        continue
        
        return sorted(transactions, key=lambda x: x['date'], reverse=True)

    # Rest of the code remains the same

    def get_total_spending(self) -> float:
        """Calculate total spending (debits)."""
        return sum(t['amount'] for t in self.transactions if t['type'] == 'DEBIT')

    def get_total_income(self) -> float:
        """Calculate total income (credits)."""
        return sum(t['amount'] for t in self.transactions if t['type'] == 'CREDIT')

    def get_balance(self) -> float:
        """Calculate net balance."""
        return self.get_total_income() - self.get_total_spending()

    def get_merchant_analysis(self) -> list:
        """Analyze spending by merchant."""
        from collections import defaultdict
        merchant_spending = defaultdict(float)
        for t in self.transactions:
            if t['type'] == 'DEBIT':
                merchant_spending[t['description']] += t['amount']
        
        # Sort by amount spent
        return sorted(merchant_spending.items(), key=lambda x: x[1], reverse=True)

def create_charts(analyzer):
    # Convert transactions to DataFrame for easier plotting
    df = pd.DataFrame([{
        'date': t['date'].strftime('%Y-%m-%d'),
        'amount': t['amount'],
        'type': t['type'],
        'description': t['description']
    } for t in analyzer.transactions])
    df['date'] = pd.to_datetime(df['date'])
    
    # Daily spending chart
    daily_spending = df[df['type'] == 'DEBIT'].groupby('date')['amount'].sum().reset_index()
    fig_daily = px.line(daily_spending, x='date', y='amount',
                       title='Daily Spending Trend',
                       labels={'amount': 'Amount (₹)', 'date': 'Date'})
    
    # Top merchants chart
    merchant_data = pd.DataFrame(analyzer.get_merchant_analysis(), 
                               columns=['merchant', 'amount'])
    fig_merchants = px.bar(merchant_data.head(10), x='merchant', y='amount',
                          title='Top 10 Merchants by Spending',
                          labels={'amount': 'Amount (₹)', 'merchant': 'Merchant'})
    fig_merchants.update_layout(xaxis_tickangle=-45)
    
    # Transaction type distribution
    fig_pie = px.pie(df, values='amount', names='type',
                     title='Transaction Distribution')
    
    return fig_daily, fig_merchants, fig_pie

def main():
    st.set_page_config(page_title="Transaction Analyzer", layout="wide")
    st.title("📊 Transaction Analyzer")
    st.write("Upload your PDF statement to analyze transactions")

    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        analyzer = TransactionAnalyzer(pdf_path)

        tab1, tab2, tab3 = st.tabs(["📈 Overview", "💰 Transactions", "📊 Charts"])

        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Spending", f"₹{analyzer.get_total_spending():,.2f}")
            with col2:
                st.metric("Total Income", f"₹{analyzer.get_total_income():,.2f}")
            with col3:
                st.metric("Net Balance", f"₹{analyzer.get_balance():,.2f}")

            if analyzer.transactions:
                st.info(f"Analysis Period: {analyzer.transactions[-1]['date'].strftime('%b %d')} to {analyzer.transactions[0]['date'].strftime('%b %d, %Y')}")

        with tab2:
            if analyzer.transactions:
                transactions_df = pd.DataFrame([{
                    'Date': t['date'].strftime('%Y-%m-%d'),
                    'Description': t['description'],
                    'Type': t['type'],
                    'Amount': f"₹{t['amount']:,.2f}"
                } for t in analyzer.transactions])

                st.dataframe(transactions_df, use_container_width=True)

                csv = transactions_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Transactions CSV",
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