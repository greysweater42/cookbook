---
title: "git"
date: 2018-12-03T15:01:35+01:00
draft: false
image: "git.jpg"
categories: ["DevOps"]
tags: ["git", "DevOps"]
---

## 1. What is git and why would you use it?

* Git is a totally basic program if you think seriously about programming. Seriously.

* It's a version control system, which makes:

    * working on the same project with many people simple;

    * remembers the whole history of the projects, i.e. all it's chages as long as you follow git's discipline

## 2. A "Hello world" example

As you may have noticed, my posts usually contain a section called 'A "Hello World" example', but not this time. There are so many tutorials and books available on the Internet that I am sure you will find something suitable for yourself very quickly.

## 3. '6 commands'

Knowing 6 git commands is a humorous description of a basic knowledge of git. But there is much truth in it: you only need 6 commands to work pretty efficiently with git. These are:

* git add

* git commit

* git clone

* git push/pull

* git merge

* git checkout


If you know all of them, you can update your LinkedIn profile with "knowledge of git". If not, I recommend youtube tutorials, like this one:

<iframe width="853" height="480" src="https://www.youtube.com/embed/HVsySz-h9r4" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## 4. Slightly more advanced subjects

#### git rebase

In it's consequences, `git rebase` is equivalent to merge, but there are certain differences:

* rebase changes the order of commits - in merge, they are chronological, in rebase - commits from branch 1 go first, then commits from branch 2;

* in merge, you usually checkout to master and run `git merge dev`, in rebase you checkout to dev and run `git rebase master`;

* DON'T REBASE PUBLIC BRANCHES, unless you want to die in pain :)

In general, when  you work on a specific project with your colleagues, I recommed using rebase, as chronological order is not that important. Thanks to rebase you can scroll the repo log and see the next functionalities (branches) appearing in order. If you even decide to give them special tags, boy, it really helps to kepp order!

Here are a few links which contain more information about rebasing: [one](https://www.atlassian.com/git/tutorials/rewriting-history/git-rebase), [two](https://benmarshall.me/git-rebase/).

* completely remove a submodule

Just replace the `<name>` with your submodule's name.

<textarea rows = "5" cols = "50" name = "git_submodule" style='font-family: "Courier New"; color: #505050;'>
submodule=<name>
git submodule deinit -f -\- $submodule
rm -rf .git/modules/$submodule
git rm -f $submodule
</textarea><br>

* remove a remote branch

Once you do rebase and merge, before pushing your changes you may want to delete the merged branch first. You can do it with:

<textarea rows = "2" cols = "50" name = "git_submodule" style='font-family: "Courier New"; color: #505050;'>
git push origin -\-delete <branch_name>
</textarea><br>

## 5. Interesting commands

Let's face the truth, after a long period of working on a project, dozens of branches appear and the repo is a complete mess. There are a few commands though, which make cleaning things up easier:

* `log`

    * `git log --follow --oneline content/git.md` - **--follow** shows log of the changes made on this particular file. **--oneline** show only a few first characters of SHA and commit message.

    * `git log --name-only --oneline` - **--name-only** shows only names of the files that were changed. Works also with git diff:

    * `git diff --name-only HEAD~2`

    * `git log --decorate` **--decorate** prints the names of all the pointers (HEAD, branches and tags) near commits SHAs. You can also use `gitk --all`. If you're not using gitka, start using it. It's good.

    * `git log --graph` - **--graph** draws commits in a form of a graph/tree.

    * `git log --all` - **--all** shows all commits, not only those on a branch you are on right now.

    * to summarise: `git log --oneline --decorate --all --graph`

* `merge` - `git checkout master; git merge develop` - **merge** branches when you are on a target branch. Seems obvious, but once I started working with rebasing and making merge requests on remote repo, I get a little confused sometimes.

* `reflog` - shows the history of everything you were doing with git in this repo: every commit, checkout, push, pull etc. Useful when you want to move back to the previous SHA you were in, before doinng checkout (*what was the name of that branch I checkout out from into master?*). Resembles `cd -` in bash.

* `stash` - can't believe I haven't mentioned *stash* by now. You can run `git stash` when you have unstaged changes in your repo, and they will be saved to a safe place until you reopen them with `git stash apply` anytime, maybe even on a different branch.


## 6. Subjects still to cover:

* git revert (TODO)

* [GitFlow (branching strategies)](https://gitversion.readthedocs.io/en/latest/git-branching-strategies/gitflow/) (TODO)

* [interactive rebase](https://www.internalpointers.com/post/squash-commits-into-one-git)
