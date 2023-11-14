# """THIS CODE WORKS WELL FOR SINGLE PAGE """
# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import statsmodels.api as sm  # Import the statsmodels library for lowess smoothing

# st.set_page_config(
#     page_title="Vector Sport Data Analysis",
#     page_icon="📈",
# )


# st.sidebar.success("Select a demo above.")

# # Define the function to convert LAP_TIME to seconds
# def lap_time_to_seconds(lap_time_str):
#     minutes, seconds = map(float, lap_time_str.split(':'))
#     return minutes * 60 + seconds

# # Streamlit app title
# st.title("Vector Sport Laps Analysis")

# # Create a file uploader widget for the 'laps' DataFrame
# csv_file = st.file_uploader("Upload Laps CSV File", type=["csv"])

# @st.cache_data
# def load_laps_data(nrows):
#     data = pd.read_csv(csv_file, nrows=nrows)
#     # If you have a 'DATE_COLUMN', you can convert it to datetime here
#     # data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data

# if csv_file is not None:
#     st.write("File uploaded successfully!")

    
#     laps = load_laps_data(10000)

#     if st.checkbox('Show raw data'):
#         st.subheader('Raw Laps data')
#         st.write(laps.head())

#         # Apply the function to the LAP_TIME column
#         laps['LAP_TIME'] = laps['LAP_TIME'].apply(lap_time_to_seconds)

#         # Use st.slider for LAP_TIME filter (same filter as the first graph)
#         lap_time_filter = st.slider("Filter by LAP_TIME", min_value=float(laps['LAP_TIME'].min()), max_value=float(laps['LAP_TIME'].max()), value=float(laps['LAP_TIME'].max() - 1))
#         filtered_laps = laps[laps['LAP_TIME'] < lap_time_filter]

#         # Filter the data for 'Vector Sport' team
#         vector_sport_laps = filtered_laps[filtered_laps['TEAM'] == 'Vector Sport']

#         # Define a dictionary to map the Team names to hex codes
#         team_colors = {'Vector Sport': '#008080'}

#         # Group the data by 'TEAM'
#         grouped = vector_sport_laps.groupby('TEAM')

#         # Convert the grouped data to a DataFrame
#         grouped_df = pd.DataFrame(grouped.apply(lambda x: x.reset_index(drop=True)))

#         # Use lmplot with the new grouped DataFrame
#         g = sns.lmplot(x='LAP_NUMBER', y='LAP_TIME', hue='TEAM', data=grouped_df, markers='o', palette=team_colors, scatter_kws={'s': 50})

#         # Set the inset background to black
#         g.ax.set_facecolor('black')

#         # Add a title
#         st.subheader('Fuel Corrected vs. Tyre Life by Team')
#         st.pyplot(plt)

#         # Add a section header for the second graph
#         st.subheader("Second Graph")

#         # Filter the data for 'Vector Sport' team (similar to the first plot)
#         vector_sport_laps_second = laps[(laps['TEAM'] == 'Vector Sport') & (laps['LAP_TIME'] < lap_time_filter)]

#         # Group the data by "Team" (for the second plot)
#         grouped_second = vector_sport_laps_second.groupby('TEAM')

#         # Convert the grouped data to a DataFrame
#         grouped_df_second = pd.DataFrame(grouped_second.apply(lambda x: x.reset_index(drop=True)))

#         # Initialize an empty list to store the slopes and team names
#         slopes = []
#         teams = []

#         # Use scatterplot with the new grouped DataFrame (for the second plot)
#         plt.figure(facecolor='black')
#         sns.scatterplot(x='LAP_NUMBER', y='LAP_TIME', hue='TEAM', data=grouped_df_second, markers='o', s=50)

#         # Calculate the slope using the lowess regression line
#         for level in vector_sport_laps_second['TEAM'].unique():
#             x = vector_sport_laps_second[vector_sport_laps_second['TEAM'] == level]['LAP_NUMBER']
#             y = vector_sport_laps_second[vector_sport_laps_second['TEAM'] == level]['LAP_TIME']
#             lowess = sm.nonparametric.lowess(y, x, frac=0.3)  # adjust the 'frac' parameter to control the smoothing
#             slope = (lowess[-1, 1] - lowess[0, 1]) / (lowess[-1, 0] - lowess[0, 0])
#             slopes.append(slope)
#             teams.append(level)
#             plt.plot(lowess[:, 0], lowess[:, 1], linewidth=2)

#             # Print the team name and slope
#             st.write('TEAM:', level, 'Slope:', slope)

#         # Customize legend text color and remove legend outline
#         legend = plt.legend(title='Team', loc='upper left', frameon=False)
#         for text in legend.get_texts():
#             text.set_color("white")

#         # Remove the box around the entire plot
#         plt.box(False)

#         # Set x and y axis labels to white
#         plt.xlabel('LAP_NUMBER', color='white')
#         plt.ylabel('LAP_TIME', color='white')

#         # Set the color of the axis ticks to white
#         plt.xticks(color='white')
#         plt.yticks(color='white')

#         # Show the plot
#         st.pyplot(plt)

#         # Print the slopes and team names
#         st.write('Teams:', teams)
#         st.write('Slopes:', slopes)

# # Optionally, you can add more widgets to customize the data analysis further, such as date filters or other user interactions.
# """END OF WORKING CODE"""


import streamlit as st
# import seaborn as sns
# import matplotlib.pyplot as plt
# import statsmodels.api as sm  # Import the statsmodels library for lowess smoothing
import os

# Streamlit app title
st.title("Vector Sport Laps Analysis")


# Define the page selection
page = st.sidebar.selectbox("Select a Page", ["Home", "Data Analysis", "Monte Carlo Simulation"])

if page == "Home":
    # st.header("Welcome to the Vector Sport Laps Analysis App")

    # Display the title
    # st.subheader("Home Page")

    # Define the path to your local video file (replace 'your_video_filename.mp4' with your actual file)
    video_path = "./VECTOR.mp4"


    # Check if the video file exists in the specified path
    if os.path.exists(video_path):
        # Display the video from your desktop
        st.video(video_path, start_time= 0)
    else:
        st.write("Video file not found. Please make sure the video file is in the specified path.")

# Check if the current page is the home page
if page == "Home":
    # Set the background image of the home page
    st.markdown(
        f"""
        <style>
            body {{
                background-image: url('./code_wavey.jpg');
                background-size: cover;
                background-position: center;
                color: white; /* Set text color to white for better visibility on a dark background */
            }}
        </style>
        """,
        unsafe_allow_html=True
    )


elif page == "Data Analysis":
    st.header("Data Analysis")

    # Your data analysis code remains the same as before, excluding the file upload part
        # Create a file uploader widget for the 'laps' DataFrame
    csv_file = st.file_uploader("Upload Laps CSV File", type=["csv"])

    @st.cache_data
    def load_laps_data(nrows):
        data = pd.read_csv(csv_file, nrows=nrows)
        # If you have a 'DATE_COLUMN', you can convert it to datetime here
        # data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
        return data

    if csv_file is not None:
            st.write("File uploaded successfully!")
            import streamlit as st
            import seaborn as sns
            import matplotlib.pyplot as plt
            import statsmodels.api as sm  # Import the statsmodels library for lowess smoothing
            # import os
            import pandas as pd


            data_load_state = st.text('Loading data...')
            laps = load_laps_data(10000)

            if st.checkbox('Show raw data'):
                st.subheader('Raw Laps data')
                st.write(laps.head())

                # Define the function to convert LAP_TIME to seconds
            def lap_time_to_seconds(lap_time_str):
                minutes, seconds = map(float, lap_time_str.split(':'))
                return minutes * 60 + seconds

            # Apply the function to the LAP_TIME column
            laps['LAP_TIME'] = laps['LAP_TIME'].apply(lap_time_to_seconds)

            # Use st.slider for LAP_TIME filter
            lap_time_filter = st.slider("Filter by LAP_TIME", min_value=float(laps['LAP_TIME'].min()), max_value=float(laps['LAP_TIME'].max()), value=float(laps['LAP_TIME'].max() - 1))
            filtered_laps = laps[laps['LAP_TIME'] < lap_time_filter]

            # Filter the data for 'Vector Sport' team
            vector_sport_laps = filtered_laps[filtered_laps['TEAM'] == 'Vector Sport']

            # Define a dictionary to map the Team names to hex codes
            team_colors = {'Vector Sport': '#008080'}

            # Group the data by 'TEAM'
            grouped = vector_sport_laps.groupby('TEAM')

            # Convert the grouped data to a DataFrame
            grouped_df = pd.DataFrame(grouped.apply(lambda x: x.reset_index(drop=True)))

            # Use lmplot with the new grouped DataFrame
            g = sns.lmplot(x='LAP_NUMBER', y='LAP_TIME', hue='TEAM', data=grouped_df, markers='o', palette=team_colors, scatter_kws={'s': 50})

            # Set the inset background to black
            g.ax.set_facecolor('black')

            # Add a title
            st.subheader('Fuel Corrected vs. Tyre Life by Team')
            st.pyplot(plt)

            # Add a section header for the second graph
            st.subheader("Second Graph")

            # Filter the data for 'Vector Sport' team (similar to the first plot)
            vector_sport_laps_second = laps[(laps['TEAM'] == 'Vector Sport') & (laps['LAP_TIME'] < lap_time_filter)]

            # Group the data by "Team" (for the second plot)
            grouped_second = vector_sport_laps_second.groupby('TEAM')

            # Convert the grouped data to a DataFrame
            grouped_df_second = pd.DataFrame(grouped_second.apply(lambda x: x.reset_index(drop=True)))

            # Initialize an empty list to store the slopes and team names
            slopes = []
            teams = []

            # Use scatterplot with the new grouped DataFrame (for the second plot)
            plt.figure(facecolor='black')
            sns.scatterplot(x='LAP_NUMBER', y='LAP_TIME', hue='TEAM', data=grouped_df_second, markers='o', s=50)

            # Calculate the slope using the lowess regression line
            for level in vector_sport_laps_second['TEAM'].unique():
                x = vector_sport_laps_second[vector_sport_laps_second['TEAM'] == level]['LAP_NUMBER']
                y = vector_sport_laps_second[vector_sport_laps_second['TEAM'] == level]['LAP_TIME']
                lowess = sm.nonparametric.lowess(y, x, frac=0.3)  # adjust the 'frac' parameter to control the smoothing
                slope = (lowess[-1, 1] - lowess[0, 1]) / (lowess[-1, 0] - lowess[0, 0])
                slopes.append(slope)
                teams.append(level)
                plt.plot(lowess[:, 0], lowess[:, 1], linewidth=2)

                # Print the team name and slope
                st.write('TEAM:', level, 'Slope:', slope)

            # Customize legend text color and remove legend outline
            legend = plt.legend(title='Team', loc='upper left', frameon=False)
            for text in legend.get_texts():
                text.set_color("white")

            # Remove the box around the entire plot
            plt.box(False)

            # Set x and y axis labels to white
            plt.xlabel('LAP_NUMBER', color='white')
            plt.ylabel('LAP_TIME', color='white')

            # Set the color of the axis ticks to white
            plt.xticks(color='white')
            plt.yticks(color='white')

            # Show the plot
            st.pyplot(plt)

            # Print the slopes and team names
            st.write('Teams:', teams)
            st.write('Slopes:', slopes)


elif page == "Monte Carlo Simulation":
    st.header("Monte Carlo Simulation")
    # st.write("This is the Monte Carlo Simulation page.")
    # Add your Monte Carlo Simulation code here


    # import streamlit as st
    import random
    import itertools
    import numpy as np
    # import matplotlib.pyplot as plt

    # Create a Streamlit app
    # st.title("Monte Carlo Simulation")
    st.header("Enter Your Tyre Compound Parameters")

    new_pace = st.number_input("New Tyre Pace", min_value=0.0, value=86.9)
    new_degradation = st.number_input("New Tyre Degradation", min_value=0.0, value=0.012)
    scrubbed_pace = st.number_input("Scrubbed Tyre Pace", min_value=0.0, value=115.81)
    scrubbed_degradation = st.number_input("Scrubbed Tyre Degradation", min_value=0.0, value=0.025)
    used_pace = st.number_input("Used Tyre Pace", min_value=0.0, value=202.7)
    used_degradation = st.number_input("Used Tyre Degradation", min_value=0.0, value=0.044)

    tyres = {
        'New': {'pace': new_pace, 'degradation': new_degradation},
        'Scrubbed': {'pace': scrubbed_pace, 'degradation': scrubbed_degradation},
        'Used': {'pace': used_pace, 'degradation': used_degradation},
    }

    # Add a "Run Simulation" button
    if st.button("Run Simulation"):
        # Number of simulations
        n_simulations = st.number_input("Number of Simulations", min_value=1, value=1000)

        # Define function to simulate race
        def simulate_race(tyre_combination):
            lap_times = []
            lap_degradation = []
            total_time = 0
            pit_stop_loss = (len(tyre_combination) - 1) * 81.8  # pit stop loss for each tyre combination
            for tyre in itertools.cycle(tyre_combination):
                pace = tyre['pace']
                degradation = tyre['degradation']
                lap_time = random.normalvariate(pace, pace * degradation)
                lap_times.append(lap_time)
                lap_degradation.append(degradation)
                total_time += lap_time
                if len(lap_times) == 245:
                    break
            total_time += pit_stop_loss  # add pit stop loss to total time
            return lap_times, lap_degradation, total_time

        # Define the 'tyre_combinations' list
        tyre_combinations = [['Scrubbed', 'Used', 'Scrubbed'], ['New', 'Used', 'Scrubbed'], ['New', 'Scrubbed', 'New'],
                            ['Scrubbed', 'Used', 'New'], ["New", "Scrubbed", "Used"], ["Scrubbed", "Used", "New"],
                            ["Scrubbed", "New", "Used"], ["Used", "New", "Scrubbed"], ["Used", "Scrubbed", "New"],
                            ["New", "New", "Used"], ["New", "New", "Scrubbed"], ["Scrubbed", "Scrubbed", "Used"],
                            ["Scrubbed", "Scrubbed", "New"], ["Used", "Used", "New"], ["Used", "Used", "Scrubbed"]]

        results = {}
        for tyre_combination in tyre_combinations:
            tyre_combination = tuple([compound for compound in tyre_combination])
            lap_times, lap_degradation, total_time = zip(
                *[simulate_race([tyres[compound] for compound in tyre_combination]) for _ in range(n_simulations)])
            results[tyre_combination] = {
                'lap_times': lap_times,
                'lap_degradation': lap_degradation,
                'total_time': total_time
            }

        # Calculate averages
        averages = {}
        for tyre_combination, data in results.items():
            averages[tyre_combination] = {
                'avg_lap_time': np.mean(data['lap_times']),
                'avg_lap_degradation': np.mean(data['lap_degradation']),
                'avg_total_time': np.mean(data['total_time'])
            }

        # Get the top 3 fastest strategies
        sorted_strategies = sorted(averages.items(), key=lambda x: x[1]['avg_total_time'])[:3]

        st.header("Top 3 Fastest Strategies")
        for i, (tyre_combination, data) in enumerate(sorted_strategies):
            st.write(f"{i + 1}. Tyre combination: {tyre_combination}")
            st.write(f"Average lap time: {data['avg_lap_time']}")
            st.write(f"Average lap degradation: {data['avg_lap_degradation']}")
            st.write(f"Average total time: {data['avg_total_time']}")

        st.header("Top 3 Fastest Strategies")
        strategy_names = []
        strategy_times = []

        for i, (tyre_combination, data) in enumerate(sorted_strategies):
            strategy_names.append(f"Strategy {i + 1}")
            strategy_times.append(data['avg_total_time'])

        # Plot the top 3 fastest strategies
        plt.figure(figsize=(8, 6))
        plt.bar(strategy_names, strategy_times)
        plt.xlabel("Strategy")
        plt.ylabel("Average Total Time")
        plt.title("Top 3 Fastest Strategies")
        st.pyplot(plt)


