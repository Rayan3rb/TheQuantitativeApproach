import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
from streamlit_extras.mention import mention


def main():

    st.markdown("<h1 style='text-align: center; color: #335575;'>Quantitative Approach for Stocks & Portfolio Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #335575;'>🌀<a href='https://twitter.com/RayanArab7' target='_blank'>Rayan Arab</a></p>", unsafe_allow_html=True)
    st.markdown("""
    Dive into a comprehensive analysis of your selected stocks:
    - Get a detailed <span style="color: #5C7791; font-weight: bold">Back-testing</span> and <span style="color: #5C7791; font-weight: bold">Statistical Analysis</span>.
    - Discover how your stocks <span style="color: #5C7791; font-weight: bold">Correlate</span> with each other.
    - Visualize the performance of an <span style="color: #5C7791; font-weight: bold">Equally Weighted Portfolio</span>.
    - Optimize your portfolio for <span style="color: #5C7791; font-weight: bold">Better Returns</span> with lesser risk.

    Simply enter the stock tickers and starting date, then press <span style="color: #5C7791; font-weight: bold">Analyze</span> to start your financial journey. Enjoy!
    """, unsafe_allow_html=True)

    # User input for stock ticker and start date
    stocks = st.text_input('Enter Stock Tickers (comma-separated):', value='2222.SR,SPUS,TSLA,GLD').upper()
    start_date = st.date_input('Enter Starting Date', pd.to_datetime('2022-12-30'))
 
    # Check if tickers are valid
    tickers = stocks.split(',')
    invalid_tickers = []
    for ticker in tickers:
        try:
            data = yf.download(stocks,start=start_date)
        except:
            invalid_tickers.append(ticker)

    if invalid_tickers:
        st.error(f"***The following tickers cannot be found: {', '.join(invalid_tickers)}***")
 
    # Analysis
    else:
        if st.button("Analyze"):
            #Calling Historical Data
            historical = data['Adj Close'].reset_index().copy()
            historical['Date'] = historical['Date'].dt.strftime('%Y/%m/%d')
            historical = historical.set_index("Date")
            historical = historical.dropna()
            # Convert DataFrame to Excel and let the user download it
            towrite = io.BytesIO()
            historical.to_excel(towrite, index=True, engine='openpyxl')
            towrite.seek(0)  
            st.markdown("<p style='color: #335575;'>Guess what, smarty? 🌟 Here's the data just for you!↙️</p>", unsafe_allow_html=True)
            st.download_button(
            label="Download the historical",
            data=towrite,
            file_name=stocks+"-historical_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
            daily_returns = historical.pct_change().dropna()
            st.write(historical)
            
            tutorial1 = mention(
            label="Find out how to do this code in Python🐍. it is an Arabic tutorial ",
            icon="https://seeklogo.com/images/T/twitter-x-logo-0339F999CF-seeklogo.com.png?v=638264860180000000", 
            url="https://x.com/RayanArab7/status/1573597230426193920?s=20",
            write=False,
            )
            coffee = mention(
            label="Feelin' the vibes from this? 🤘 Grab me a coffee!",
            icon="https://upload.wikimedia.org/wikipedia/en/5/5c/Buy_Me_a_Coffee_Logo.png",
            url="https://www.buymeacoffee.com/rayan3rab7",
            write=False,
            )
            colab1 = mention(
            label="Lets see how i did the code with Python. Check it out, play with it, and learn something cool!🌟",
            icon="https://pbs.twimg.com/profile_images/1330956917951270912/DyIZtTA8_400x400.png",
            url="https://colab.research.google.com/drive/1JnOkKW1IcXQciFlyWk9uQTi0bQF8aaby?usp=drive_link",
            write=False,
            )
            st.write(tutorial1,unsafe_allow_html=True)
            st.write(colab1,unsafe_allow_html=True)
            st.write(coffee,unsafe_allow_html=True)

            #------------------------------------------------------------------------------------
            # Buy and Hold Backtesting for each ticker
            st.markdown("<h2 style='text-align: left; color: #5C7791;'>💹 Historical Buy & Hold Back-Testing</h1>", unsafe_allow_html=True)
            st.markdown("<p style='color: #335575;'>Let's play pretend! If you started with <b>$1000</b>💸 and picked these tickers, here's how your money would've grown for each ticker 🌱📈</p>", unsafe_allow_html=True)
            dollarvalue = 1000
            historical_line = historical.pct_change().dropna()
            historical_line.iloc[0, :] = 1 * dollarvalue - 1
            historical_line = (historical_line + 1).cumprod()
            st.line_chart(historical_line, use_container_width=True)
            colab2 = mention(
            label="Do you wanna see how i did the code in Python?🌟",
            icon="https://pbs.twimg.com/profile_images/1330956917951270912/DyIZtTA8_400x400.png",
            url="https://colab.research.google.com/drive/1cJY7T4hoNgxgsSZn5PjoQSHQ4qyT_L79?ouid=106099351402784250382&usp=drive_link",
            write=False,
            )
            st.write(colab2,unsafe_allow_html=True)

            #------------------------------------------------------------------------------------
            # Seperate Line
            st.markdown("<hr style='border:4px solid lightgrey'>", unsafe_allow_html=True)
            #------------------------------------------------------------------------------------
            # Statistical analysis for each ticker
            st.markdown("<h2 style='text-align: left; color: #5C7791;'>🔢 Statistical Analysis</h1>", unsafe_allow_html=True)

            trading_days = 252
            today_mp = np.round(historical_line.iloc[-1],2)
            hp_return = np.round((historical_line.iloc[-1]/historical_line.iloc[0]-1)*100,2)
            arithmetic_mean = np.round(daily_returns.mean()*trading_days*100,4)
            geometric_mean = np.round((((np.prod(1 + daily_returns))**(trading_days/len(daily_returns))) - 1)*100,4)
            standard_deviation = np.round(daily_returns.std()* np.sqrt(trading_days)*100,4)

            metrics = pd.concat([today_mp,hp_return,arithmetic_mean,geometric_mean,standard_deviation],axis=1)
            metrics.columns = ["Today Value","Holding Return %",'Arithmetic Mean %', 'Geometric Mean %', 'Standard Deviation %']

            st.write(metrics)
            with st.expander("👨‍🏫Expalin the table for me plz💤"):
                 st.markdown("""
                <style>
                .custom-text {
                    color: #335575;
                }st
                </style>
                <div class="custom-text">
                    <b>Today's Value:</b> It's like checking your land value when you want to sell it. Did the price increase or decrease? 🛩️🪂<br>
                    <b>Holding Return:</b> It’s about tracking the growth (or dip) of your money over time. Think of it as monitoring your GPA. 📊📈<br>
                    <b>Annualized Arithmetic Mean:</b> This is the average daily growth rate of your investment. Imagine tracking your daily study hours, then finding the average. 📚⏰<br>
                    <b>Annualized Geometric Mean:</b> If you were to spread out your investment's growth consistently across the year, this tells you its pace. Kinda like evenly spacing out your study sessions before finals. 📅📝<br>
                    <b>Annualized Standard Deviation:</b> Investments can be unpredictable. This tells you how wild or calm the ride was. 🎢📉
                </div>
                """, unsafe_allow_html=True)
            tutorial2 = mention(
            label="Find out how to do this code in Python🐍. it is an Arabic tutorial ",
            icon="https://seeklogo.com/images/T/twitter-x-logo-0339F999CF-seeklogo.com.png?v=638264860180000000",  # Twitter is also featured!
            url="https://x.com/RayanArab7/status/1671876587019722752?s=20",
            write=False,
            )
            colab3 = mention(
            label="i have the Python code here just for you 😁 No need to thank me, one cup of coffee will be lovely☕",
            icon="https://pbs.twimg.com/profile_images/1330956917951270912/DyIZtTA8_400x400.png",
            url="https://colab.research.google.com/drive/1qpRfRUHg9IwvDrEp0IADTojhVHAPZugq?ouid=106099351402784250382&usp=drive_link",
            write=False,
            )
            st.write(tutorial2,unsafe_allow_html=True)
            st.write(colab3,unsafe_allow_html=True)            
            st.write(coffee,unsafe_allow_html=True)

            #-----------------------------------------------------------------------------------------------
            # Seperate Line
            st.markdown("<hr style='border:4px solid lightgrey'>", unsafe_allow_html=True)
            #-----------------------------------------------------------------------------------------------
            # Technical Analysis
            st.markdown("<h2 style='text-align: left; color: #5C7791;'>📊 Technical Analysis</h1>", unsafe_allow_html=True)
            def compute_rsi(data, window=14):
                delta = data.diff()
                up, down = delta.clip(lower=0), -1*delta.clip(upper=0)
                ema_up = up.ewm(com=window, adjust=False).mean()
                ema_down = down.ewm(com=window, adjust=False).mean()
                rs = ema_up/ema_down
                rsi = 100 - (100 / (1 + rs))
                latest_rsi = np.round(rsi.tail(1), 2)
                signal = ['Overbought' if value > 70 else 'Oversold' if value < 30 else 'Neutral' for value in latest_rsi]
                latest_rsi['RSI Signal'] = signal
                latest_rsi.index = ['RSI Value', 'RSI Signal']
                return latest_rsi
            def compute_stochastic(data, window=14):
                high14 = data.rolling(window=window).max()
                low14 = data.rolling(window=window).min()
                k_percent = 100 * ((data - low14) / (high14 - low14))
                d_percent = k_percent.rolling(window=3).mean()
                latest_k = np.round(k_percent.tail(1), 2)
                latest_d = np.round(d_percent.tail(1), 2)
                signal = ['Bullish' if k > d else 'Bearish' for k, d in zip(latest_k, latest_d)]
                latest_k['Stochastic Signal'] = signal
                latest_k.index = ['%K Value', 'Stochastic Signal']
                return latest_k
            def compute_macd(data, short_window=12, long_window=26, signal_window=9):
                short_ema = data.ewm(span=short_window, adjust=False).mean()
                long_ema = data.ewm(span=long_window, adjust=False).mean()
                macd = short_ema - long_ema
                signal = macd.ewm(span=signal_window, adjust=False).mean()
                latest_macd = np.round(macd.tail(1), 2)
                latest_signal = np.round(signal.tail(1), 2)
                signal_text = ['Bullish' if m > s else 'Bearish' for m, s in zip(latest_macd, latest_signal)]
                latest_macd['MACD Signal'] = signal_text
                latest_macd.index = ['MACD Value', 'MACD Signal']
                return latest_macd
            historical_rsi = historical.apply(compute_rsi)
            historical_stochastic = historical.apply(compute_stochastic)
            historical_macd = historical.apply(compute_macd)
            combined_indicators = pd.concat([historical_rsi, historical_stochastic,historical_macd], axis=0).T
            st.write(combined_indicators)
            with st.expander("Learn more about these indicators! 🎓✨"):
                st.write("""
                ### RSI (Relative Strength Index)
                Think of RSI as the mood ring of the stock market! It tells us if a stock is feeling overbought (too many buyers) or oversold (too many sellers). Values over 70 are like a stock's way of saying, "I'm getting too hot!" (Overbought) and under 30, "I'm feeling chilly!" (Oversold). Anywhere in between, and the stock's just chillin'. 😎

                ### Stochastic Oscillator
                This is like a game of musical chairs for stocks! 🎶 The Stochastic Oscillator compares where a stock's price is relative to its price range over a certain period. If it's closer to the highest price, the music's about to stop, and it might be time to sell. Closer to the lowest? Grab that chair and consider buying!

                ### MACD (Moving Average Convergence Divergence)
                Sounds complex, right? But it's just two moving buddies going for a walk - a short-term moving average and a long-term one. 🚶‍♂️🚶‍♀️ When the short-term buddy walks faster and goes ahead (crosses above the long-term), it's a bullish sign!📈 This means the market is expected to rise. When it lags behind (crosses below), it's bearish.📉 This implies the market might see a downturn. The difference between the two? That's our MACD line!
                """)
            #-----------------------------------------------------------------------------------------------
            # Seperate Line
            st.markdown("<hr style='border:4px solid lightgrey'>", unsafe_allow_html=True)
            #-----------------------------------------------------------------------------------------------
            # Correlation explanations and take away
            st.markdown("<h2 style='text-align: left; color: #5C7791;'>🔄 Stocks Correlation</h1>", unsafe_allow_html=True)

            corr = daily_returns.corr()
            fig, ax = plt.subplots(figsize=(10, 7))
            from matplotlib.colors import LinearSegmentedColormap
            colors = ["#335575", "#BDC7D2"]
            cmap_custom = LinearSegmentedColormap.from_list("blue_to_navy", colors)
            sns.heatmap(corr, annot=True, cmap=cmap_custom, vmin=-1, vmax=1, ax=ax, annot_kws={"size": 12}) 
            st.pyplot(fig)
            with st.expander("👨‍🏫Expalin Correlation for me plz💤"):
                st.markdown("""
                <div style="color: #335575;">

                🌙 **Stock Correlation: A Tale of Two Camels in the Desert!** 🐪🐪

                Let's embark on a desert journey with two camels. How these camels tread the sands is a lot like how stocks behave in the market:

                - **Positive Correlation:** Our two camels walk step-by-step, in perfect sync. If one speeds up, the other does too; if one takes a break, the other rests beside it. This is just like two stocks that tend to go up or down together. If one stock is having a good day, chances are the other is too! 🐪🐪

                - **Negative Correlation:** Imagine one camel decides to rest under a palm tree while the other feels adventurous and wanders off. They're doing the complete opposite of each other! Similarly, when one stock rises, the other tends to fall. It's like they're playing a seesaw game in the market. 🐪🌴

                - **Zero Correlation:** Here, our camels have a mind of their own. One might be grazing while the other is strolling. They're independent and don't really affect each other's path. In the stock world, this means the movement of one stock doesn't give us any clue about the movement of the other.

                </div>
                """, unsafe_allow_html=True)
            for i in range(corr.shape[0]):
                for j in range(i+1, corr.shape[1]):
                    value = corr.iloc[i, j]
                    stock1 = corr.index[i]
                    stock2 = corr.columns[j]
                    if value > 0.5:
                        st.markdown(f"📈 {stock1} and {stock2} have a **high positive correlation** of {value:.2f}. They tend to move in the same direction.")
                    elif value < -0.5:
                        st.markdown(f"📉 {stock1} and {stock2} have a **high negative correlation** of {value:.2f}. They tend to move in opposite directions, which can be great for diversification.")
            colab4 = mention(
            label="Do you know that in one line of code you can do the Correlation in Python🤯",
            icon="https://pbs.twimg.com/profile_images/1330956917951270912/DyIZtTA8_400x400.png",
            url="https://colab.research.google.com/drive/1J-Eqlha9PGKBYIkN7nDgm-DIVG_pwikG?ouid=106099351402784250382&usp=drive_link",
            write=False,
            )
            st.write(colab4,unsafe_allow_html=True)
            
            #-----------------------------------------------------------------------------------------------    
            # Seperate Line
            st.markdown("<hr style='border:4px solid lightgrey'>", unsafe_allow_html=True)
            #-----------------------------------------------------------------------------------------------
            
            # Equal Weight Porfolio
            st.markdown("<h2 style='text-align: left; color: #5C7791;'>⚖️ Equally Weighted Portfolio</h1>", unsafe_allow_html=True)

            st.markdown(f"**What if** you Invested in this tickers with equal weights")
            cov = np.round(daily_returns.cov(),5)

            weight = [1/len(tickers)] * len(tickers)

            custom_colors = [cmap_custom(i/len(tickers)) for i in range(len(tickers))]
            fig, ax = plt.subplots()
            ax.pie(weight, labels=tickers, autopct='%1.1f%%', startangle=45, colors=custom_colors)
            ax.axis('equal')
            st.pyplot(fig)

            portfolio_returns = daily_returns
            portfolio_returns.iloc[0,:] = (np.array(weight) * dollarvalue)-1
            portfolio_cum_returns = (portfolio_returns + 1).cumprod()
            portfolio_cum_returns['Portfolio'] = portfolio_cum_returns.sum(axis=1)
            portfolio = portfolio_cum_returns['Portfolio']
            st.markdown("### The Performance would be ↗️")

            st.line_chart(portfolio, use_container_width=True)
            
            porfolio_today_mp = np.round(portfolio.tail(1),2)
            porfolio_hp_return = np.round((porfolio_today_mp/portfolio.iloc[0]-1)*100,2)
            porfolio_arithmetic_mean = np.round(portfolio.pct_change().mean()*trading_days*100,4)
            porfolio_geometric_mean = np.round((((np.prod(1 + portfolio.pct_change()))**(trading_days/len(portfolio.pct_change()))) - 1)*100,4)
            porfolio_standard_deviation = np.round(portfolio.pct_change().std()* np.sqrt(trading_days)*100,4)
            porfolio_metrics_data = {
                "Metrics": ["Today Value", "Holding Return %", 'Arithmetic Mean %', 'Geometric Mean %', 'Standard Deviation %'],
                "Values": [porfolio_today_mp.values[0], porfolio_hp_return.values[0], porfolio_arithmetic_mean, porfolio_geometric_mean, porfolio_standard_deviation]
            }
            porfolio_metrics = pd.DataFrame(porfolio_metrics_data).set_index('Metrics').T
            
            st.write(porfolio_metrics)
            colab5 = mention(
            label="A great skill to learn is to code portfolio back-testing with Python, got your back😉",
            icon="https://pbs.twimg.com/profile_images/1330956917951270912/DyIZtTA8_400x400.png",
            url="https://colab.research.google.com/drive/1J-Eqlha9PGKBYIkN7nDgm-DIVG_pwikG?ouid=106099351402784250382&usp=drive_link",
            write=False,
            )
            st.write(colab5,unsafe_allow_html=True)

            #-----------------------------------------------------------------------------------------------
            # Seperate Line
            st.markdown("<hr style='border:4px solid lightgrey'>", unsafe_allow_html=True)
            #-----------------------------------------------------------------------------------------------
            # Optimized Portfolio
            st.markdown("<h2 style='text-align: left; color: #5C7791;'>🤖 Optimized Portfolio</h1>", unsafe_allow_html=True)
            st.markdown(f"""
            🔮✨ By diving into the magic of the Modern Portfolio Theory, we're setting sails to optimize risk while cruising close to those geometric returns! 🚀🌌 Talk about a win-win, right? 🥳🍾
            """, unsafe_allow_html=True)
            N = len(daily_returns.columns)

            def calculate_returns(weights):
                return np.sum(np.round(np.prod(1 + historical.pct_change())**(trading_days/len(historical.pct_change())) - 1,4)*weights)

            from scipy.optimize import minimize
            equal_weights = N * [1/N]
            bound = [(0,1) for _ in daily_returns.columns]
            constraint = ({'type':'eq','fun': lambda weights: np.sum(weights)-1},
                             {'type':'eq','fun': lambda weights: calculate_returns(weights)- np.round(int(np.sum(metrics['Geometric Mean %'] * weight)),0)/100})
            def calculate_volatility(weights):
                annualized_cov = np.dot(cov,weights)
                return np.sqrt(np.dot(weights.transpose(),annualized_cov))
            # Minimized STD
            result = minimize(calculate_volatility, x0=np.round(equal_weights,3), bounds = bound, constraints = constraint)
            opt =pd.DataFrame(result['x'], index=daily_returns.columns, columns=['Optimized Weights'])
            #######################################################################################################################################################            
            fig, ax = plt.subplots()
            ax.pie(opt['Optimized Weights'], labels=daily_returns.columns, autopct='%1.1f%%', startangle=45, colors=custom_colors)
            st.pyplot(fig)

            optimized_returns = daily_returns
            optimized_returns.iloc[0,:] = (opt['Optimized Weights'] * dollarvalue)-1
            optimized_cum_returns = (optimized_returns + 1).cumprod()
            optimized_cum_returns['optimized'] = optimized_cum_returns.sum(axis=1)
            optimized = optimized_cum_returns['optimized']

            st.markdown("### The Optimized Performance would be ↗️")
            st.line_chart(optimized, use_container_width=True)

            optimized_today_mp = np.round(optimized.tail(1),2)
            optimized_hp_return = np.round((optimized_today_mp/optimized.iloc[0]-1)*100,2)
            optimized_arithmetic_mean = np.round(optimized.pct_change().mean()*trading_days*100,4)
            optimized_geometric_mean = np.round((((np.prod(1 + optimized.pct_change()))**(trading_days/len(optimized.pct_change()))) - 1)*100,4)
            optimized_standard_deviation = np.round(optimized.pct_change().std()* np.sqrt(trading_days)*100,4)

            optimized_metrics_data = {
                "Metrics": ["Today Value", "Holding Return %", 'Arithmetic Mean %', 'Geometric Mean %', 'Standard Deviation %'],
                "Values": [optimized_today_mp.values[0], optimized_hp_return.values[0], optimized_arithmetic_mean, optimized_geometric_mean, optimized_standard_deviation]
            }
            optimized_metrics_data = pd.DataFrame(optimized_metrics_data).set_index('Metrics').T
            st.write(optimized_metrics_data)            
            if optimized_metrics_data['Geometric Mean %'].values > porfolio_metrics['Geometric Mean %'].values:
                st.markdown(f"""
                🚀 Boom! Our optimized Geometric Mean just soared higher by <span style='color:green'>**{float(np.round(optimized_metrics_data['Geometric Mean %'] - porfolio_metrics['Geometric Mean %'],2))}**</span> 
                Hold on! The ride's getting smoother, too 🎢 the Standard Deviation dropped down by <span style='color:red'>**{float(np.round(optimized_metrics_data['Standard Deviation %'] - porfolio_metrics['Standard Deviation %'],2))}**</span>. Yeah baby, that's what I'm talking about - more return with low risk!
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"🎉 Surprise! We managed to smooth out the ride by lowering the Standard Deviation by **{float(np.round(optimized_metrics_data['Standard Deviation %'] - porfolio_metrics['Standard Deviation %'],2))}**. But, oh snap! Our returns took a tiny nap. Guess it's the classic 'less risk less returns' deal! 🎢")
                
            st.markdown("""
                        

            ### 🙌 Thank You for Using The Quantitative Approach for Stocks & Portfolio Analysis!
            If you found this analysis useful and learned something new, consider supporting the effort behind it. A cup of coffee would be perfect! ☕️ 
            Don't forget to share this tool with your friends and fellow investors! Spread the knowledge and happy investing! 🌟
            """, unsafe_allow_html=True)
            from streamlit_extras.buy_me_a_coffee import button

            button(username="rayan3rab7", floating=False, width=221)

if __name__ == '__main__':
    main()