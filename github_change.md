1）写自己的功能/改动

以后不要在 main 上改了，统一在自己的分支改：

git checkout my-changes
# 写代码...
git add -A
git commit -m "some feature"
git push


要新功能可以从 main 再拉一个新分支，例如：

git checkout main
git pull origin main
git checkout -b feature/xxx
# 在 feature/xxx 上写代码

2）想要同步上游的更新时

例如过几天上游有更新了，你想把新代码合进来：

先更新 main：

git checkout main
git fetch upstream
git merge upstream/main      # 或者：git reset --hard upstream/main 再 push
git push origin main


再把最新 main 合并到你的工作分支，例如 my-changes：

git checkout my-changes
git merge main               # 有冲突就解决冲突，git add，git commit
git push


或者你喜欢更干净的历史，可以用 rebase：

git checkout my-changes
git rebase main
# 解决冲突 -> git add 冲突文件
git rebase --continue
git push --force-with-lease
