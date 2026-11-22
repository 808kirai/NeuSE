Deploying NeuSE

There are two ways to deploy it.

- On any computer with Docker, in about ten minutes, to try it or to check the results. See the first part below
- On Huawei Cloud, with HTTPS and a managed database, so field units anywhere can sync to it. See the second part

The Docker image is built and run, and a sighting sent to it and read back, on every push to the repository, so both ways start from an image that is known to work.

Part 1. On any computer with Docker

This runs the same command center image as the cloud deployment, on your own computer, with field units flown against it from the HIT-UAV test images.

What you need

- Docker, and Git
- Python 3.12, for the demo units
- About 1.3 GB of disk for the HIT-UAV dataset

1. Get the code and the data

```bash
git clone https://github.com/808kirai/NeuSE.git
cd NeuSE
pip install -r requirements.txt
git clone --depth 1 https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset.git ../datasets/hit-uav
```

2. Build and start the command center

Make a token first. The command center will not start on a network with a token shorter than 16 characters.

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
docker build -f deploy/Dockerfile -t neuse-command-center .
docker run -d --name neuse -p 8000:8000 -e NEUSE_CC_TOKEN=PASTE-THE-TOKEN neuse-command-center
curl http://127.0.0.1:8000/api/v1/health
```

The last command answers `{"status":"ok"}` when it is running.

3. Open the map

Open http://127.0.0.1:8000 in a browser and enter the token. The map is empty until units report.

4. Fly three units against it

```bash
python scripts/demo_units.py --frames-dir ../datasets/hit-uav/normal_json/test --units 3 --frames 120 --center http://127.0.0.1:8000 --token PASTE-THE-TOKEN
```

Units appear on the map within seconds, then their sightings, the ground they have searched and any alerts. Links drop at random and the backlog arrives when they come back. Press Write report for a situation report.

5. Check the detector's scores

```bash
python scripts/evaluate_hit_uav.py --dataset ../datasets/hit-uav --split test
python -m pytest -q tests
```

The first gives mAP@0.5 0.670 on the 579 test images, as reported in the README. The second runs every automated test.

6. Stop it

```bash
docker rm -f neuse
```

Without Docker, `NEUSE_CC_TOKEN=PASTE-THE-TOKEN python -m command_center.app` runs the same command center straight from the code, on http://127.0.0.1:8000.

Part 2. On Huawei Cloud

This puts the command center on the internet behind HTTPS, so field units anywhere can sync to it and coordinators can open the map. It uses an ECS server for the command center and RDS for MySQL for its database. Huawei Cloud vouchers are given to teams that reach the competition's regional stage, so these steps will be run then. The image and the MySQL support are checked automatically on every push.

What you need

- A Huawei Cloud account with credit or vouchers
- About an hour
- A domain name pointing at the server, or the free sslip.io name described in step 4
- This repository and a terminal with SSH

1. Network

In the Huawei Cloud console, in the region you will use for everything, create a VPC with one subnet. Create a security group for the server with these inbound rules.

- SSH, port 22, from your own IP address only
- HTTP, port 80, from anywhere, used to get the HTTPS certificate
- HTTPS, port 443, from anywhere

Create a second security group for the database that allows MySQL, port 3306, only from the server's security group.

2. Database, RDS for MySQL

Create an RDS for MySQL 8.0 instance in the same VPC and subnet, with the database security group. The smallest single instance is enough for a demo. When it is ready, create a database called `neuse` and an account called `neuse` with full rights on it, and note the instance's private IP address.

Leaving out RDS also works. The command center then keeps an SQLite file on the server, which is fine for a demo but is lost if the server is deleted.

3. Server, ECS

Create an ECS instance running Ubuntu 24.04 with 2 vCPUs and 4 GB of memory, in the same VPC, with the server security group and an elastic IP address. Log in with SSH and install Docker and Git.

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2 git
```

4. Domain

Point a domain name at the elastic IP address, for example `neuse.example.org`. With no domain of your own, use sslip.io, which answers with the address written in the name. For a server at 1.2.3.4 the name is `1-2-3-4.sslip.io`.

5. Settings

```bash
git clone https://github.com/808kirai/NeuSE.git
cd NeuSE/deploy
cp .env.example .env
nano .env
```

Fill in `.env`. Its comments explain each line.

- `NEUSE_DOMAIN` - The name from step 4
- `NEUSE_CC_TOKEN` - The coordinator token, a long random string
- `NEUSE_CC_UNIT_TOKENS` - One token per field unit
- `NEUSE_CC_DATABASE_URL` - `mysql://neuse:PASSWORD@PRIVATE-IP:3306/neuse` with the RDS details from step 2

Make each token with `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`. In the database address, write `@` in the password as `%40` and `:` as `%3A`.

6. Start it

```bash
sudo docker compose up -d --build
sudo docker compose logs -f command-center
```

Caddy gets the HTTPS certificate by itself the first time someone opens the address, which can take a minute. Check it answers.

```bash
curl https://neuse.example.org/api/v1/health
```

Open `https://neuse.example.org` in a browser and enter the coordinator token.

7. Field units

On each field unit, sync to the new address with that unit's own token.

```bash
NEUSE_UNIT_ID=unit-1 NEUSE_SYNC_TOKEN=unit-1-token python -m src.sync --url https://neuse.example.org
```

To fill the map for a demo, fly the demo units from any computer with the HIT-UAV test images. The coordinator token can send for any unit.

```bash
python scripts/demo_units.py --frames-dir ../datasets/hit-uav/normal_json/test --units 3 --center https://neuse.example.org --token COORDINATOR-TOKEN
```

8. Language model for situation reports

In ModelArts Studio, open a chat model that is available in your region, create an API key, and copy the model's API address and model name. Put them in `.env`. The address is the part before `/chat/completions`.

- `NEUSE_LLM_URL` - The API address
- `NEUSE_LLM_KEY` - The API key
- `NEUSE_LLM_MODEL` - The model name

Then restart with `sudo docker compose up -d`. The next report written on the map says whether it came from the model or the template.

9. Text alerts through SMN

The command center raises an alert when one area, sightings within 100 m of each other, gets at least 3 sightings with a confidence of 0.8 or more that reach the command center within 10 minutes. A backlog from a unit that was out of range counts when it arrives, as long as the sightings are under 2 hours old. Alerts always show on the map. With Simple Message Notification set up, each one also goes out as a text message, which reaches a coordinator's phone on any network, even 2G. An area that was just alerted stays quiet for 30 minutes so phones are not flooded.

- In the SMN console, in the same region as the server, create a topic called neuse-alerts
- Add a subscription to the topic for each coordinator's phone, with the protocol set to SMS. Each phone gets a message asking to confirm
- Copy the topic's URN
- Under My Credentials, copy the project ID for the region, and create an access key. An IAM user that can only publish to SMN is safer than the main account's key
- Put them in `.env`

- `NEUSE_SMN_TOPIC_URN` - The topic's URN
- `NEUSE_SMN_REGION` - The region, such as `af-south-1`
- `NEUSE_SMN_PROJECT_ID` - The project ID for that region
- `CLOUD_SDK_AK` and `CLOUD_SDK_SK` - The access key and secret key, the names Huawei Cloud's own SDK uses
- `NEUSE_PUBLIC_URL` - The address coordinators open, added to each text

`NEUSE_ALERT_MIN_SIGHTINGS`, `NEUSE_ALERT_MIN_CONFIDENCE`, `NEUSE_ALERT_WINDOW_MINUTES`, `NEUSE_ALERT_MAX_AGE_MINUTES` and `NEUSE_ALERT_QUIET_MINUTES` change when alerts are raised.

Restart with `sudo docker compose up -d`. The alerts list on the map says whether texts are going out, and an alert whose text failed shows why. Text messages are charged per message, so check the SMN price for your region.

10. Security

- The command center will not start with a token shorter than 16 characters or one still holding the example value. Make each with `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- A caller that gives 20 wrong tokens in 5 minutes is turned away for the rest of that window. The Docker setup passes on each caller's own address from Caddy, so one caller cannot lock out the others
- Every API request is logged with who made it, without the token. Read the log with `sudo docker compose logs command-center`
- Set `NEUSE_CC_KEEP_DAYS` in `.env` to remove old sightings, searched ground and alerts once a search is over
- To revoke a lost unit, remove its entry from `NEUSE_CC_UNIT_TOKENS` and restart with `sudo docker compose up -d`
- Open only ports 80 and 443 in the security group, keep RDS on the private network, and give the SMN access key to an IAM user that can only publish
- `docs/SAFETY.md` has the full list of measures and the risks that remain

Updating

```bash
cd NeuSE
git pull
cd deploy
sudo docker compose up -d --build
```

When something goes wrong

- The page does not load - Check ports 80 and 443 are open in the server's security group and that the domain points at the elastic IP address
- The certificate fails - Caddy's log says why, `sudo docker compose logs https`. Let's Encrypt needs port 80 open and the domain pointing at the server
- The command center stops at start - Its log says why, `sudo docker compose logs command-center`. A wrong database address or password shows up here
- Units cannot send - The unit's log shows the reply. 401 means a wrong token, and a unit refuses to send its token over plain HTTP to another computer

Costs

ECS and RDS charge while they exist, so stop or delete them when the demo is over. Check current prices in the Huawei Cloud console before creating them.
